# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from pyskl import __version__
from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port

###
import numpy as np
import decord
from tqdm import tqdm
from pyskl.smp import mrlines
###

try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')


def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]

def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame)
        results.append(result)
    return results

def pose_inference(model, frames, det_results):
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
    return kp

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    
    # * Both mmdet and mmpose should be installed from source

    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-ckpt',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-ckpt',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')


    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos',
                        default='examples/extract_diving48_skeleton/diving48.list')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name',
                         default='examples/extract_diving48_skeleton/diving48_annos.pkl')
    parser.add_argument('--tmpdir', type=str, default='tmp')

    ####
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)


    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)


    # -------------------------------------------------
    lines = mrlines(args.video_list)
    lines = [x.split() for x in lines]

    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines]
    else:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]


    if rank == 0:
        os.makedirs(args.tmpdir, exist_ok=True)
    dist.barrier()
    my_part = annos[rank::world_size]


    # ----------------------------------------------------------------------
    print(f'init detection model')
    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')

    print(f'init pose model')
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda')

    # ----------------------------------------------------------------------

    print(f'start')
    for anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        det_results = detection_inference(det_model, frames)
        # * Get detection results for human
        det_results = [x[0] for x in det_results]
        for i, res in enumerate(det_results):
            # * filter boxes with small scores
            res = res[res[:, 4] >= args.det_score_thr]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            # assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res

        pose_results = pose_inference(pose_model, frames, det_results)
        shape = frames[0].shape[:2]
        anno['img_shape'] = anno['original_shape'] = shape
        anno['total_frames'] = len(frames)
        anno['num_person_raw'] = pose_results.shape[0]
        anno['keypoint'] = pose_results[..., :2].astype(np.float16)
        anno['keypoint_score'] = pose_results[..., 2].astype(np.float16)
        anno.pop('filename')

        print(f'0000')
        
    print("almost done")
    mmcv.dump(my_part, osp.join(args.tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank == 0:
        parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
        rem = len(annos) % world_size
        if rem:
            for i in range(rem, world_size):
                parts[i].append(None)

        ordered_results = []
        for res in zip(*parts):
            ordered_results.extend(list(res))
        ordered_results = ordered_results[:len(annos)]
        mmcv.dump(ordered_results, args.out)

if __name__ == '__main__':
    main()
