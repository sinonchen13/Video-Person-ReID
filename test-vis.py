from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import argparse
import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import tools.data_manager as data_manager
from tools.video_loader import VideoDataset
from tools.utils import Logger
from tools.metrics_vis import evaluate as evaluate_vis

parser = argparse.ArgumentParser(description='Test using all frames')
# Datasets
parser.add_argument('--root', type=str, default='/media/sdb1/zzj/datasets')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--test_sample_mode', type=str, default='rrs0',
                    help="test_all_sampled, rrs0")
# Augment
parser.add_argument('--test_frames', default=4, type=int,
                    help='frames per clip for test')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='baseline',
                    help="baseline, ame")
# Miscs
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--test_epochs', default=[240], nargs='+', type=int)
parser.add_argument('--distance', type=str, default='cosine',
                    help="euclidean or cosine")
parser.add_argument('--gpu', default='0, 1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def get_actmap(features, sz,seq_len):
    """
    :param features: (1, 2048, 16, 8) activation map
    :return:
    """
    #print(features.size())
    features = (features ** 2).sum(1)  # (1, 16, 8)
    bt, h, w = features.size()
    features = features.view(bt, h * w)
    features = F.normalize(features, p=2, dim=1)
    acts = features.view(bt, h, w)
    b=bt//seq_len
    all_acts = []
    for i in range(b):
        tmp=[]
        for j in range(seq_len):
            act = acts[i*seq_len+j].numpy()
            act = cv2.resize(act, (sz[1], sz[0]))
            act = 255 * (act - act.min()) / (act.max() - act.min() + 1e-12)  #act = 255 * (act - act.max()) / (act.max() - act.min() + 1e-12)
            act = np.uint8(np.floor(act))
            act = cv2.applyColorMap(act, cv2.COLORMAP_JET)
            tmp.append(act)
        all_acts.append(tmp)
    return all_acts


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

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

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(
        name=args.arch, num_classes=dataset.num_train_pids)
    print("Model size: {:.5f}M".format(sum(p.numel()
          for p in model.parameters())/1000000.0))

    for epoch in args.test_epochs:
        model_path = osp.join(
            args.resume, 'checkpoint_ep'+str(epoch)+'.pth.tar')
        # model_path = osp.join(args.resume, 'best_model.pth.tar')
        print("Loading checkpoint from '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict(checkpoint['state_dict'])

        if use_gpu:
            model = model.cuda()

        print("Evaluate")
        with torch.no_grad():
            test(model, queryloader, galleryloader, use_gpu)


def extract(model, vids, use_gpu):
    n, c, f, h, w = vids.size()

    assert(n == 1)

    if use_gpu:
        feat = torch.cuda.FloatTensor()
    else:
        feat = torch.FloatTensor()
    for i in range(math.ceil(f/args.test_frames)):
        clip = vids[:, :, i*args.test_frames:(i+1)*args.test_frames, :, :]
        if use_gpu:
            clip = clip.cuda()

        act_outputs=[]
        def hook_fns_forward(module, input, output):
            act_outputs.append(output.cpu())
        handle = model.backbone.register_forward_hook(hook_fns_forward)
        output = model(clip)
        handle.remove()
        # 默认seq_len 4帧
        acts = get_actmap(act_outputs[0],[128,64],4)
        feat = torch.cat((feat, output), 2)
    feat = feat.mean(2)
    feat = model.bn(feat)
    feat = feat.data.cpu()

    return feat,acts


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


def test(model, queryloader, galleryloader, use_gpu):
    since = time.time()
    model.eval()

    
    qf, q_pids, q_camids, q_paths ,q_act= [], [], [], [],[]

    for batch_idx, (vids, pids, camids,img_paths) in enumerate(queryloader):
        if (batch_idx + 1) % 1000 == 0:
            print("{}/{}".format(batch_idx+1, len(queryloader)))
        fs,acts=extract(model, vids, use_gpu)
        qf.append(fs.squeeze())
        q_pids.extend(pids)
        q_camids.extend(camids)
        q_paths.append(np.asarray(img_paths).transpose())
        q_act.extend(acts)


    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_paths = np.concatenate(q_paths)

    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids, g_paths,g_act = [], [], [], [],[]
    for batch_idx, (vids, pids, camids,img_paths) in enumerate(galleryloader):
        if (batch_idx + 1) % 1000 == 0:
            print("{}/{}".format(batch_idx+1, len(galleryloader)))
        fs,acts=extract(model, vids, use_gpu)
        gf.append(fs.squeeze())
        g_pids.extend(pids)
        g_camids.extend(camids)
        g_paths.append(np.asarray(img_paths).transpose())
        g_act.extend(acts)

    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    g_paths = np.concatenate(g_paths)


    if args.dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)
        g_paths = np.concatenate([q_paths,g_paths]) 
        g_act = np.concatenate([q_act,g_act])

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))

    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        for i in range(m):
            # ori  - torch.mm(qf[i:i+1], gf.t())
            distmat[i] = 1 - torch.mm(qf[i:i+1], gf.t())

    # save dist for ensemble
    distmat = distmat.numpy()
    np.save('distmat_'+args.arch, distmat)
    
    
    re_distmat = cam_add_distmat(distmat, q_camids, g_camids)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate_vis(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths, q_act=q_act,g_act=g_act,plot_ranking=True)

    print("ori Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format( cmc[0], cmc[4], cmc[9], mAP))
    print("------------------")
    cmc, mAP = evaluate_vis(re_distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths, q_act=q_act,g_act=g_act,plot_ranking=False)
    print("re Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    return cmc[0]


if __name__ == '__main__':
    main()
