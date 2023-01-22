import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models import build_network, model_fn_decorator
from torch.autograd import Variable

from pcdet.datasets.processor import data_processor

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


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
    # iter_eps = args.eps/np.sqrt(3)/30
    nb_iter = 20
    # rand_init = True
    # eps = args.eps/np.sqrt(3) # 0.3
    # norm = np.inf
    # decay_factor = 1
    # clip_min = None
    # clip_max = None
    iter_ratio = args.ratio/nb_iter
    

    point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    if key == 'voxels' or 'pv_rcnn' in args.cfg_file:
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

    for i, batch_dict in enumerate(dataloader):
        # import pdb;pdb.set_trace()
        load_data_to_gpu(batch_dict)
        # batch_dict['points'].require_grad = True
        # points_origin = batch_dict['points'].clone()
        voxels_origin = batch_dict[key].clone()
        voxels_origin.require_grad = False # important

        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

        # print("### batch_dict voxels shape", batch_dict[key].shape)
        batch_dict[key].requires_grad = True
        
        for i in range(nb_iter):
            for cur_module in model.module_list:
                # print("## iterate", cur_module)
                # import pdb;pdb.set_trace()
                batch_dict = cur_module(batch_dict)
            # loss, tb_dict = model.dense_head.get_loss()
            loss, tb_dict, _ = model.get_training_loss()

            model.zero_grad()
            batch_dict[key].retain_grad()
            # loss.backward() # 
            loss.backward(retain_graph=True)
            grad = batch_dict[key].grad.data

            # remove voxels/points according to the grad
            # import pdb;pdb.set_trace()
            ## remove voxels/points directly
            batch_dict[key].requires_grad = False
            if key=='voxels':

                grad[batch_dict[key]==0] = 0 #important, put the value of padded part in voxels = 0
                grad_sum = torch.sum(torch.abs(grad), axis=2).flatten()

                valid_points_num = (grad_sum!=0).sum()
                if i == 0:
                    iter_remove_num = int(args.ratio / nb_iter * valid_points_num)
                values, indices = grad_sum.topk(iter_remove_num, largest=True)
                # indices = torch.randint(valid_points_num, (int(iter_ratio*valid_points_num),)) # random, should remove valid, please note

                voxels_flatten = batch_dict[key].view(-1, num_point_features)
                with torch.no_grad():
                    # import pdb;pdb.set_trace()
                    # voxels_flatten[grad_sum.nonzero()[indices]] = 0 # for random remove
                    voxels_flatten[indices] = 0
                    # todo #######voxel_num_points
                # 1. ori
                # batch_dict[key] = voxels_flatten.view(batch_dict[key].shape) # ad hoc 5
                # 2. revoxelize, unneccessary because no points are shifted
                points_sum = voxels_flatten.sum(1)
                points_valid = voxels_flatten[points_sum!=0].cpu().numpy()
                voxels, coordinates, num_points = voxel_generator.generate(points_valid)
                batch_dict['voxels'] = torch.from_numpy(voxels).cuda(voxels_flatten.device)
                pad_batch_indexs = np.zeros((len(voxels),1))
                coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                batch_dict['voxel_coords'] = torch.from_numpy(coordinates).cuda(voxels_flatten.device)
                batch_dict['voxel_num_points'] = torch.from_numpy(num_points).cuda(voxels_flatten.device)

            else:
                iter_remove_num = int(args.ratio / nb_iter * len(voxels_origin))
                grad_sum = torch.sum(torch.abs(grad), 1)
                values, indices = grad_sum.topk(len(voxels_origin) - iter_remove_num * (i+1), largest=False)
                batch_dict[key] = batch_dict[key][indices]

                if 'pv_rcnn' in args.cfg_file:
                    voxels, coordinates, num_points = voxel_generator.generate(batch_dict[key][:, 1:].cpu().numpy())
                    batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).cuda(voxels_origin.device)
                    pad_batch_indexs = np.zeros((len(voxels),1))
                    coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
                    batch_dict['voxel_coords'] = torch.from_numpy(coordinates).cuda(voxels_origin.device)
                    batch_dict['voxel_num_points'] = torch.from_numpy(num_points).cuda(voxels_origin.device)

            
            batch_dict[key].requires_grad = True

            for k in ['batch_index', 'point_cls_scores', 'batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized', 'rois', 'roi_scores', 'roi_labels', 'has_class_labels']:
                batch_dict.pop(k, None) # adhoc

        model.eval()
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
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

    parser.add_argument('--ratio', type=float, default=0.1, help='the ratio of points to be removed')
    # parser.add_argument('--attack', type=str, default='PGD', help='FGSM/PGD/MI')
    parser.add_argument('--key', type=str, default='voxels', help='voxels/points')
    # parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')

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


