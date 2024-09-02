import glob
import os

import torch
import tqdm
import time
import numpy as np
import copy
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from pcdet.datasets.processor import data_processor

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


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()
        loss_clean, tb_dict, disp_dict = model_func(model, batch)

        key = 'voxels'
        # key = 'points'
        model_name = 'second'

        if True:
            rand_init = True
            iter_eps = 0.05/1
            nb_iter = 1        
            eps = 0.05
            norm = 2 # np.inf 2
        else:
            rand_init = False
            nb_iter = 0

        if key == 'voxels':
            # max_num_points_per_voxel = 32
            max_num_points_per_voxel = 5
            max_num_voxels = 16000 # training
            # voxel_size = [0.16, 0.16, 4]
            voxel_size = [0.05, 0.05, 0.1]
            # point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
            point_cloud_range = [0, -40, -3, 70.4, 40, 1]
            num_point_features=4
            voxel_generator = data_processor.VoxelGeneratorWrapper(
                vsize_xyz=voxel_size,
                coors_range_xyz=point_cloud_range,
                num_point_features=num_point_features+1,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels,
            )
        elif model_name == 'pv_rcnn':
            max_num_points_per_voxel = 5
            max_num_voxels = 16000 # training
            voxel_size = [0.05, 0.05, 0.1]
            point_cloud_range = [0, -40, -3, 70.4, 40, 1]
            num_point_features=4
            voxel_generator = data_processor.VoxelGeneratorWrapper(
                vsize_xyz=voxel_size,
                coors_range_xyz=point_cloud_range,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels,
            )
        
        batch_dict = batch
        if key == 'voxels':
            points_flatten = batch_dict[key].view(-1, num_point_features)
            points_sum = (points_flatten.abs()).sum(1)
            key_origin = points_flatten[points_sum!=0].clone()
            points_origin = key_origin.cpu().numpy()
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

        for m in model.module.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
        batch_dict[key].requires_grad = True

        for i in range(nb_iter):
            # print("### iteration", i)
            for cur_module in model.module.module_list:
                # print("## iterate", cur_module)
                batch_dict = cur_module(batch_dict)
            # loss, tb_dict = model.dense_head.get_loss()
            loss, _, _ = model.module.get_training_loss()

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
            # if 'second' in args.cfg_file or 'voxel_rcnn' in args.cfg_file or 'PartA2' in args.cfg_file:
            if model_name in ['second', 'voxel_rcnn', 'PartA2']:
                grad = - grad # for second

            perturbation = clip_eta(grad, iter_eps, norm)

            batch_dict[key].requires_grad = False
            if key == 'voxels':
                batch_dict[key][:, :, :3] = batch_dict[key][:, :, :3] + perturbation

                # if args.attack != 'MI' or (args.attack == 'MI' and i==nb_iter-1):
                if True: # for PGD
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
                perturbation = clip_eta(perturbation, eps, norm)
                perturbated_point_coords = key_origin + perturbation
                batch_dict[key][:, 1:4] = perturbated_point_coords

                # if 'pv_rcnn' in args.cfg_file:
                if model_name == 'pv_rcnn':
                    voxels, coordinates, num_points = voxel_generator.generate(batch_dict[key][:, 1:].cpu().numpy())
                    batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).float().cuda(key_origin.device)
                    pad_batch_indexs = np.zeros((len(voxels),1))
                    coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                    batch_dict['voxel_coords'] = torch.from_numpy(coordinates).float().cuda(key_origin.device)
                    batch_dict['voxel_num_points'] = torch.from_numpy(num_points).float().cuda(key_origin.device)


            batch_dict[key].requires_grad = True

            for k in ['batch_index', 'point_cls_scores', 'batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized', 'rois', 'roi_scores', 'roi_labels', 'has_class_labels']:
                batch_dict.pop(k, None) # adhoc

            for k in ['voxel_features', 'encoded_spconv_tensor', 'encoded_spconv_tensor_stride', 'multi_scale_3d_features', 'multi_scale_3d_strides', 'spatial_features', 'spatial_features_stride', 'spatial_features_2d']:
                batch_dict.pop(k, None)

        batch_dict[key].requires_grad = False
        model.train()
        optimizer.zero_grad()
        loss_adv, tb_dict_adv, disp_dict_adv = model_func(model, batch)

        gamma = 2
        p_clean, p_adv = torch.softmax(torch.tensor([-loss_clean.detach(), -loss_adv.detach()]), dim=0)
        focal_clean = - (1 - p_clean) ** gamma * torch.log(p_clean) + 0.5
        focal_adv = - (1 - p_adv) ** gamma * torch.log(p_adv) + 0.5

        # weight_clean = focal_clean / (focal_clean + focal_adv)
        # weight_adv = focal_adv / (focal_clean + focal_adv)
        loss = focal_clean * loss_clean + focal_adv * loss_adv

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'loss_clean': loss_clean.item(), 'loss_adv': loss_adv.item(),
                'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
