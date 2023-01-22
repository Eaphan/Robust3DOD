import _init_path
import argparse
import datetime
import glob
import os
import re
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
import tqdm

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models import build_network, model_fn_decorator
from torch.autograd import Variable
from pcdet.datasets.processor import data_processor

from sklearn.cluster import KMeans
import pandas as pd

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def clip_eta(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """
    grad_shape = grad.shape
    grad_shape_len = len(grad.shape)
    if grad_shape_len == 3:
        grad = grad.view(-1, 3)

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-36, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.sum(grad ** 2, red_ind, keepdim=True)
        optimal_perturbation = grad / torch.max(torch.sqrt(square), avoid_zero_div)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    if grad_shape_len == 3:
        scaled_perturbation = scaled_perturbation.view(grad_shape)
    return scaled_perturbation

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, args, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    # define some hyper-parameters
    key = args.key
    iter_eps = args.eps/30
    nb_iter = 40
    rand_init = True
    eps = args.eps # 0.3
    norm = 2 # np.inf 2
    decay_factor = 1
    clip_min = None
    clip_max = None
    model_func=model_fn_decorator()
    point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    if key == 'voxels':
        max_num_points_per_voxel = [x['MAX_POINTS_PER_VOXEL'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]
        max_num_voxels = [x['MAX_NUMBER_OF_VOXELS'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]['test']
        voxel_size = [x['VOXEL_SIZE'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]

        num_point_features=4 if 'kitti' in args.cfg_file else 5
        voxel_generator = data_processor.VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features+1,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels,
        )
    elif 'pv_rcnn' in args.cfg_file:
        max_num_points_per_voxel = [x['MAX_POINTS_PER_VOXEL'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]
        max_num_voxels = [x['MAX_NUMBER_OF_VOXELS'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]['test']
        voxel_size = [x['VOXEL_SIZE'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]

        num_point_features=4 if 'kitti' in args.cfg_file else 5
        voxel_generator = data_processor.VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels,
        )

    if args.attack == 'FGSM': # FGSM
        iter_eps = args.eps
        rand_init = False
        nb_iter = 1
    if args.attack == 'MI':
        rand_init = False

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        # batch_dict['points'].requires_grad = True
        # points_origin = batch_dict['points'].clone()
        if key == 'voxels':
            # key_origin = batch_dict[key][:, :, :3].clone()
            points_flatten = batch_dict[key].view(-1, num_point_features)
            points_sum = (points_flatten.abs()).sum(1)
            key_origin = points_flatten[points_sum!=0].clone()
            points_origin = key_origin.cpu().numpy()
            # print("### ori valid points num = ", len(points_origin))
        else:
            key_origin = batch_dict[key][:, 1:4].clone()
        key_origin.requires_grad = False # important

        if rand_init:
            # perturbation = torch.zeros_like(points_origin[:, 1:5]).uniform_(-eps, eps).cuda(points_origin.device)
            if key=='voxels':
                perturbation = torch.zeros_like(key_origin[:, :3]).uniform_(-eps, eps).cuda(key_origin.device)
            else:
                perturbation = torch.zeros_like(key_origin).uniform_(-eps, eps).cuda(key_origin.device)
            perturbation = clip_eta(perturbation, eps, norm)

            if key == 'voxels':
                points_valid = copy.deepcopy(points_origin)
                points_valid[:, :3] = points_valid[:, :3] + perturbation.cpu().numpy()

                # re-voxelize
                # points_flatten = batch_dict[key].view(-1, num_point_features)
                # points_flatten[points_flatten[:, 2]>=1, 2] = 1-1e-6
                # points_sum = points_flatten.sum(1)
                # points_valid = points_flatten[points_sum!=0].cpu().numpy()
                points_valid[points_valid[:, 0]>=point_cloud_range[3], 0] = point_cloud_range[3] - 1e-6
                points_valid[points_valid[:, 1]>=point_cloud_range[4], 1] = point_cloud_range[4] - 1e-6
                points_valid[points_valid[:, 2]>=point_cloud_range[5], 2] = point_cloud_range[5] - 1e-6

                points_valid[points_valid[:, 0]<point_cloud_range[0], 0] = point_cloud_range[0]
                points_valid[points_valid[:, 1]<point_cloud_range[1], 1] = point_cloud_range[1]
                points_valid[points_valid[:, 2]<point_cloud_range[2], 2] = point_cloud_range[2]

                points_valid = np.concatenate([points_valid, np.arange(len(points_valid)).reshape(-1,1)], axis=1)
                voxels, coordinates, num_points = voxel_generator.generate(points_valid)

                batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).float().cuda(key_origin.device)
                # TODO: get the index of item in voxels
                pad_batch_indexs = np.zeros((len(voxels),1))
                coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                batch_dict['voxel_coords'] = torch.from_numpy(coordinates).float().cuda(key_origin.device)
                batch_dict['voxel_num_points'] = torch.from_numpy(num_points).float().cuda(key_origin.device)
            else:
                batch_dict[key][:, 1:4] = batch_dict[key][:, 1:4] + perturbation
                if 'pv_rcnn' in args.cfg_file:
                    voxels, coordinates, num_points = voxel_generator.generate(batch_dict[key][:, 1:].cpu().numpy())
                    batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).float().cuda(key_origin.device)
                    pad_batch_indexs = np.zeros((len(voxels),1))
                    coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                    batch_dict['voxel_coords'] = torch.from_numpy(coordinates).float().cuda(key_origin.device)
                    batch_dict['voxel_num_points'] = torch.from_numpy(num_points).float().cuda(key_origin.device)
        else:
            if key=='voxels':
                # points_with_index = np.concatenate([points_origin, np.arange(len(points_origin)).reshape(-1,1)], axis=1)
                # voxels, coordinates, num_points = voxel_generator.generate(points_with_index)
                voxel_points_index = torch.zeros(batch_dict['voxels'].shape[0] * batch_dict['voxels'].shape[1], device=key_origin.device)
                valid_points_index = torch.arange(len(key_origin), device=key_origin.device)
                try:
                    voxel_points_index[torch.sum(batch_dict['voxels'].abs(), dim=2).flatten().nonzero().flatten()] = valid_points_index.float()
                except:
                    import pdb;pdb.set_trace()
                voxels = torch.cat([batch_dict[key], voxel_points_index.view(batch_dict['voxels'].shape[0], batch_dict['voxels'].shape[1], 1)], axis=2)
                voxels = voxels.cpu().numpy()
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

        # print("### batch_dict voxels shape", batch_dict[key].shape)
        batch_dict[key].requires_grad = True
        
        if key=='voxels':
            g = torch.zeros_like(batch_dict[key][:, :, :3]).to(key_origin.device)
        else:
            g = torch.zeros_like(key_origin).to(key_origin.device)

        for i in range(nb_iter):
            # print("### iteration", i)
            for cur_module in model.module_list:
                # print("## iterate", cur_module)
                batch_dict = cur_module(batch_dict)
            # loss, tb_dict = model.dense_head.get_loss()
            loss, tb_dict, _ = model.get_training_loss()

            # add distance loss of voxels and points
            # criterion = nn.MSELoss(reduction='mean')
            # if key == 'voxels':
            #     loss_distance = criterion(batch_dict[key][:, :, :3], key_origin)
            # else:
            #     loss_distance = criterion(batch_dict[key][:, 1:4], key_origin)
            # loss = loss + loss_distance

            model.zero_grad()
            batch_dict[key].retain_grad()
            loss.backward(retain_graph=True)
            grad = batch_dict[key].grad.data

            if key=='voxels':
                grad[batch_dict[key]==0] = 0
                grad = grad[:, :, :3]
            else:
                grad = grad[:, 1:4]

            # adhoc, replace the grad with random variables
            # grad_zero_mask = grad==0
            # grad = torch.zeros_like(grad).uniform_(-0.1, 0.1).cuda(key_origin.device)
            # grad[grad_zero_mask] = 0
            if 'second' in args.cfg_file or 'voxel_rcnn' in args.cfg_file or 'PartA2' in args.cfg_file:
                grad = - grad

            if args.attack == 'MI':
                g = decay_factor * g + grad/torch.norm(grad, p=1)
                perturbation = clip_eta(g, iter_eps, norm)
            else:
                perturbation = clip_eta(grad, iter_eps, norm)
                # perturbation = torch.clamp(grad, -eps, eps)

            # perturbation = - perturbation
            # perturbation = torch.zeros_like(perturbation)

            batch_dict[key].requires_grad = False
            if key == 'voxels':
                # perturbation = batch_dict[key][:, :, :3] + perturbation - key_origin
                # perturbation = clip_eta(perturbation, eps, norm)
                # batch_dict[key][:, :, :3] = key_origin + perturbation
                batch_dict[key][:, :, :3] = batch_dict[key][:, :, :3] + perturbation

                if args.attack != 'MI' or (args.attack == 'MI' and i==nb_iter-1):
                    voxels_with_pointindex = torch.cat([batch_dict[key], torch.from_numpy(voxels[:, :, -1:]).float().cuda(key_origin.device)], axis=2)
                    points_flatten = voxels_with_pointindex.view(-1, num_point_features+1)

                    points_sum = (points_flatten.abs()).sum(1)
                    points_valid = points_flatten[points_sum!=0].cpu().numpy()
                    # points_valid = points_valid[points_valid[:,4].argsort()]

                    # print("###, iter=", i)
                    # import pdb;pdb.set_trace()
                    perturbation = points_valid[:, :3] - points_origin[points_valid[:, -1].astype(int), :3]

                    ##################################################### second clip_eta
                    perturbation = clip_eta(torch.from_numpy(perturbation), eps, norm).numpy()
                    # perturbation = np.clip(perturbation, -eps, eps)
                    # print('### perturbation', perturbation)
                    points_valid[:, :3] = points_origin[points_valid[:, -1].astype(int), :3] + perturbation

                    # limit the points in the point cloud range 
                    points_valid[points_valid[:, 0]>=point_cloud_range[3], 0] = point_cloud_range[3] - 1e-6
                    points_valid[points_valid[:, 1]>=point_cloud_range[4], 1] = point_cloud_range[4] - 1e-6
                    points_valid[points_valid[:, 2]>=point_cloud_range[5], 2] = point_cloud_range[5] - 1e-6

                    points_valid[points_valid[:, 0]<point_cloud_range[0], 0] = point_cloud_range[0]
                    points_valid[points_valid[:, 1]<point_cloud_range[1], 1] = point_cloud_range[1]
                    points_valid[points_valid[:, 2]<point_cloud_range[2], 2] = point_cloud_range[2]

                    # print("### Before voxelize points_valid.shape", points_valid.shape, points_valid.max(0), points_valid.min(0))
                    voxels, coordinates, num_points = voxel_generator.generate(points_valid)

                    batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).float().cuda(key_origin.device)
                    pad_batch_indexs = np.zeros((len(voxels),1))
                    coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                    batch_dict['voxel_coords'] = torch.from_numpy(coordinates).float().cuda(key_origin.device)
                    batch_dict['voxel_num_points'] = torch.from_numpy(num_points).float().cuda(key_origin.device)

                    # print("### after perturbation and re-voxelization, valid points num = ", (points_sum!=0).sum())
            else:
                perturbation = batch_dict[key][:, 1:4] + perturbation - key_origin
                # perturbation = torch.clamp(perturbation, -eps, eps)
                perturbation = clip_eta(perturbation, eps, norm)
                perturbated_point_coords = key_origin + perturbation
                batch_dict[key][:, 1:4] = perturbated_point_coords

                if 'pv_rcnn' in args.cfg_file:
                    voxels, coordinates, num_points = voxel_generator.generate(batch_dict[key][:, 1:].cpu().numpy())
                    batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).float().cuda(key_origin.device)
                    pad_batch_indexs = np.zeros((len(voxels),1))
                    coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                    batch_dict['voxel_coords'] = torch.from_numpy(coordinates).float().cuda(key_origin.device)
                    batch_dict['voxel_num_points'] = torch.from_numpy(num_points).float().cuda(key_origin.device)


            batch_dict[key].requires_grad = True
            # import pdb;pdb.set_trace()

            # re-voxelize

            # points_flatten = batch_dict[key].view(-1, num_point_features)
            # points_sum = points_flatten.sum(1)
            # points_valid = points_flatten[points_sum!=0].cpu().numpy()
            # voxels, coordinates, num_points = voxel_generator.generate(points_valid)
            # batch_dict['voxels'] = torch.from_numpy(voxels).cuda(key_origin.device)
            # pad_batch_indexs = np.zeros((len(voxels),1))
            # coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
            # batch_dict['voxel_coords'] = torch.from_numpy(coordinates).cuda(key_origin.device)
            # batch_dict['voxel_num_points'] = torch.from_numpy(num_points).cuda(key_origin.device)

            for k in ['batch_index', 'point_cls_scores', 'batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized', 'rois', 'roi_scores', 'roi_labels', 'has_class_labels']:
                batch_dict.pop(k, None) # adhoc

            for k in ['voxel_features', 'encoded_spconv_tensor', 'encoded_spconv_tensor_stride', 'multi_scale_3d_features', 'multi_scale_3d_strides', 'spatial_features', 'spatial_features_stride', 'spatial_features_2d']:
                batch_dict.pop(k, None)

            # batch_dict['voxels'] = key_origin + torch.clamp(batch_dict['voxels'] + clip_eta(grad, iter_eps, norm) - key_origin, -eps, eps)
            # print("### newest perturbation is ", new_perturbation, i)

            # If clipping is needed, reset all values outside of [clip_min, clip_max]
            if (clip_min is not None) or (clip_max is not None):
                if clip_min is None or clip_max is None:
                    raise ValueError(
                        "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
                    )
                # batch_dict['points'][:, 1:5] = torch.clamp(batch_dict['points'][:, 1:5], clip_min, clip_max)

        ############## defense ##############
        if args.defense:
            if key=='voxels':
                # import pdb;pdb.set_trace()
                # ## random sample
                # voxels_flatten  = batch_dict[key].view(-1, 4)
                # voxels_flatten_sum = torch.sum(voxels_flatten, axis=1)
                # voxels_flatten_nonzero_num = (voxels_flatten_sum!=0).sum()
                # indices = torch.randint(voxels_flatten_nonzero_num, (int(0.1*voxels_flatten_nonzero_num),)) # random, should remove valid, please note
                # voxels_flatten[indices] = 0
                # batch_dict[key] = voxels_flatten.view(batch_dict[key].shape)

                ### add Gaussian Noises
                # noise = (0.05**0.5)*torch.randn(batch_dict[key].shape, device=batch_dict[key].device)
                # noise[batch_dict[key]==0] = 0
                # batch_dict[key] = batch_dict[key] + noise
                ### Quantification
                # batch_dict[key] = torch.round(batch_dict[key])

                ### flip, old and wrong
                # batch_dict[key][:, :, 1] = -batch_dict[key][:, :, 1]
                # batch_dict['voxel_coords'][:, 2] = 1599 - batch_dict['voxel_coords'][:, 2]

                # 1) get points
                assert args.batch_size==1
                voxels_flatten = batch_dict[key].view(-1, 4)
                points = voxels_flatten[voxels_flatten.sum(1)!=0]
                # 2) voxelization
                voxel_generator_defense = data_processor.VoxelGeneratorWrapper(
                    vsize_xyz=voxel_size,
                    coors_range_xyz=point_cloud_range,
                    num_point_features=num_point_features,
                    max_num_points_per_voxel=max_num_points_per_voxel,
                    max_num_voxels=max_num_voxels,
                )
                points = points.cpu().detach().numpy()

                if args.d_type == 'quantify':
                    # quantification
                    points[:, :3] = np.round(points[:, :3]*100) / 100.0
                elif args.d_type == 'flip':
                    ### flip points
                    points[:, 1] *= -1
                elif args.d_type == 'scale':
                    ## scale points
                    noise_scale = 1.025 # 0.95 1 1.05
                    points[:, :3] *= noise_scale
                elif args.d_type == 'rotate':
                    ## rotate points
                    noise_rotation = np.pi/8
                    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
                elif args.d_type == 'gaussian':
                    # noise = np.random.randn(*points[:, :3].shape) * (0.05**0.5)
                    noise = np.random.normal(0, 0.01, points[:, :3].shape)
                    points[:, :3] += noise
                elif args.d_type == 'sample':
                    ind = np.arange(len(points))
                    num = int(len(points) * 0.95)
                    sub_ind = np.random.choice(ind, num, replace=False)
                    points = points[sub_ind]
                elif args.d_type == 'knn':
                    # import pdb;pdb.set_trace()
                    p_coords = points[:, :3]
                    estimator = KMeans(n_clusters=20)
                    estimator.fit(p_coords)
                    label_pred = estimator.labels_
                    centroids = estimator.cluster_centers_
                    distance = np.sum(np.abs(p_coords - centroids[label_pred]) ** 2, axis=1)
                    some = np.stack([label_pred, distance], 1)
                    frame = pd.DataFrame(some, columns = ['label', 'distance'])

                    # distance_mean = np.array(frame.groupby('label')['distance'].mean())
                    distance_std = np.array(frame.groupby('label')['distance'].std())
                    mask = distance < distance_std[label_pred]*3
                    points = points[mask]
                else:
                    raise NotImplementedError()
                
                try:
                    voxels, coordinates, num_points = voxel_generator_defense.generate(points)
                except:
                    import pdb;pdb.set_trace()

                coordinates = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=0)
                voxels = torch.from_numpy(voxels).float().cuda(key_origin.device)
                coordinates = torch.from_numpy(coordinates).float().cuda(key_origin.device)
                num_points = torch.from_numpy(num_points).float().cuda(key_origin.device)
                batch_dict['voxels'] = voxels
                batch_dict['voxel_coords'] = coordinates
                batch_dict['voxel_num_points'] = num_points
            else:
                batch_dict['points'].requires_grad = False

                if args.d_type == 'quantify':
                    batch_dict['points'][:, 1:4] = torch.round(batch_dict['points'][:, 1:4]*100) / 100.0
                elif args.d_type == 'flip':
                    # flip
                    batch_dict['points'][:, 2] = - batch_dict['points'][:, 2]
                elif args.d_type == 'scale':
                    # scale
                    noise_scale=1.025 # 0.95 1 1.05
                    batch_dict['points'][:, 1:4] *= noise_scale
                elif args.d_type == 'rotate':
                    # rotate
                    noise_rotation = torch.from_numpy(np.array([np.pi/8])).float().cuda(batch_dict[key].device)
                    batch_dict['points'][:, 1:4] = common_utils.rotate_points_along_z(batch_dict['points'][:, 1:4].view(1,-1,3), noise_rotation)[0]
                elif args.d_type == 'gaussian':
                    noise = torch.empty(batch_dict['points'][:, 1:4].shape, device=batch_dict[key].device).normal_(mean=0,std=0.01)
                    # noise = (0.05**0.5)*torch.randn(batch_dict['points'][:, 1:4].shape, device=batch_dict[key].device)
                    # noise = np.random.randn(*points[:, :3].shape) * (0.05**0.5)
                    batch_dict['points'][:, 1:4] = batch_dict['points'][:, 1:4] + noise
                elif args.d_type == 'sample':
                    # ind = np.arange(len(points))
                    # num = int(len(points) * 0.95)
                    # sub_ind = np.random.choice(ind, num, replace=False)
                    # points = points[sub_ind]
                    # import pdb;pdb.set_trace()
                    points_num = batch_dict['points'].size(0)
                    perm = torch.randperm(points_num)
                    idx = perm[:int(points_num * 0.95)]
                    batch_dict['points'] = batch_dict['points'][idx]
                elif args.d_type == 'knn':
                    p_coords = batch_dict['points'][:, 1:4].cpu().numpy()
                    estimator = KMeans(n_clusters=20)
                    estimator.fit(p_coords)
                    label_pred = estimator.labels_
                    centroids = estimator.cluster_centers_
                    distance = np.sum(np.abs(p_coords - centroids[label_pred]) ** 2, axis=1)
                    some = np.stack([label_pred, distance], 1)
                    frame = pd.DataFrame(some, columns = ['label', 'distance'])

                    distance_std = np.array(frame.groupby('label')['distance'].std())
                    mask = distance < distance_std[label_pred]*3
                    batch_dict['points'] = batch_dict['points'][mask]
                else:
                    raise NotImplementedError()

                if 'pv_rcnn' in args.cfg_file:
                    voxels, coordinates, num_points = voxel_generator.generate(batch_dict[key][:, 1:].cpu().numpy())
                    batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).float().cuda(key_origin.device)
                    pad_batch_indexs = np.zeros((len(voxels),1))
                    coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                    batch_dict['voxel_coords'] = torch.from_numpy(coordinates).float().cuda(key_origin.device)
                    batch_dict['voxel_num_points'] = torch.from_numpy(num_points).float().cuda(key_origin.device)

        if args.save_points:
            attack_type = os.path.basename(__file__).split('.')[0]
            frame_id = batch_dict['frame_id'][0]
            model_name = os.path.basename(args.cfg_file).split('.')[0]
            output_dir = f'../output/kitti_models/{model_name}/{attack_type}_{args.attack}_{eps}/'
            os.makedirs(output_dir, exist_ok=True)
            output_path = f'../output/kitti_models/{model_name}/{attack_type}_{args.attack}_{eps}/{frame_id}.bin'
            if args.key == 'voxels':
                save_points = points_valid[:, :4]
            else:
                save_points = batch_dict[key][:, 1:].detach().cpu().numpy()

            with open(output_path, 'w') as f:
                save_points.tofile(f)
                # how to load: obj_points = np.fromfile(str(output_path), dtype=np.float32).reshape(-1 ,4)

        model.eval()
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None,
            flip=(args.d_type=='flip'), rotate=(args.d_type=='rotate'), scale=(args.d_type=='scale')
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # with open(result_dir / 'result.pkl', 'wb') as f:
    #     pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--eps', type=float, default=0.03, help='max_shift default 0.03m')
    parser.add_argument('--attack', type=str, default='PGD', help='FGSM/PGD/MI')
    parser.add_argument('--key', type=str, default='voxels', help='voxels/points')
    # parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--defense', action='store_true', default=False, help='')
    parser.add_argument('--d_type', default='', type=str, help='flip/rotate/scale/gaussian/quantify/sample')
    parser.add_argument('--save_points', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    # if not args.eval_all:
    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    # else:
    # eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    ## rewrite the process of evaluation
    eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, args, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


if __name__ == '__main__':
    main()




