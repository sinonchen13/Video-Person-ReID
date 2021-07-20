from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import models
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import tools.data_manager as data_manager
from tools.video_loader import VideoDataset
from tools.utils import Logger
from tools.eval_metrics import evaluate

parser = argparse.ArgumentParser(description='Test using all frames')
# Datasets
parser.add_argument('--root', type=str, default='/media/sdb1/zzj/datasets')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--test_sample_mode', type=str, default='test_all_sampled',#可视化时选择 rrs0
                    help="test_all_sampled, rrs0")
# Augment
parser.add_argument('--test_frames', default=4, type=int,
                    help='frames per clip for test')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='baseline',
                    help="baseline, ame")
# Miscs
parser.add_argument('--resume', type=str, default='./', metavar='PATH')
parser.add_argument('--test_epochs', default=[240], nargs='+', type=int)
parser.add_argument('--distance', type=str, default='cosine',
                    help="euclidean or cosine")
parser.add_argument('--gpu', default='0, 1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
#Vis
parser.add_argument('--vis', default=1, type=int,
                    help='0  False, 1 True')
args = parser.parse_args()


def cam_add_distmat(distmat, q_camids, g_camids):
    max_dist = np.max(np.max(distmat))
    q_camids_list = q_camids.tolist()
    g_camids_list = g_camids.tolist()
    cams_equals = np.zeros((len(q_camids_list), len(g_camids_list)))
    for i in range(len(q_camids_list)):
        for j in range(len(g_camids_list)):
            if q_camids_list[i] == g_camids_list[j]:
                cams_equals[i][j] = max_dist
    distmat = distmat + cams_equals
    return distmat


def main():
    #vis
    if args.vis==1:
        args.test_sample_mode="rrs0"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    #log
    sys.stdout = Logger(osp.join(args.resume, 'log_test_{}.txt'.format(args.test_sample_mode)))
    print("==========\nArgs:{}\n==========".format(args))

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    # Data augmentation
    spatial_transform_test = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    temporal_transform_test = TT.TemporalSample(mode=args.test_sample_mode, seq_len=args.test_frames)

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    test(queryloader, galleryloader, use_gpu)


def test(queryloader, galleryloader, use_gpu):
    since = time.time()

    q_pids, q_camids = [], []
    for batch_idx, (vids, pids, camids) in enumerate(queryloader):
        if (batch_idx + 1) % 1000 == 0:
            print("{}/{}".format(batch_idx+1, len(queryloader)))

        q_pids.extend(pids)
        q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    g_pids, g_camids = [], []
    for batch_idx, (vids, pids, camids) in enumerate(galleryloader):
        if (batch_idx + 1) % 1000 == 0:
            print("{}/{}".format(batch_idx+1, len(galleryloader)))

        g_pids.extend(pids)
        g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    if args.dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")

    dist_list = ["distmat_me.npy","distmat_multiloss.npy","distmat_coordatt.npy","distmat_coordatt_me_multiloss.npy"]
    
    for i in range(len(dist_list)):
        if i==0:
            distmat=np.load(dist_list[i]) 
        else:
            distmat+=np.load(dist_list[i])
    
    re_distmat = cam_add_distmat(distmat, q_camids, g_camids)
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("ori Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(
        cmc[0], cmc[4], cmc[9], mAP))
    print("------------------")
    cmc, mAP = evaluate(re_distmat, q_pids, g_pids, q_camids, g_camids)
    print("re Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))


    return cmc[0]


if __name__ == '__main__':
    main()
