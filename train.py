from __future__ import print_function, absolute_import
import os
import gc
import cv2
import sys
import time
import math
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
# from apex import amp   #########
import models
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import tools.data_manager as data_manager
from tools.video_loader import VideoDataset
from tools.losses import TripletLoss
from tools.utils import AverageMeter, Logger, save_checkpoint
from tools.eval_metrics import evaluate as evaluate_ori
from tools.metrics_vis import evaluate as evaluate_vis
from tools.samplers import RandomIdentitySampler
import torch.backends.cudnn as cudnn
print("==========Begin==========")
parser = argparse.ArgumentParser(description='Train')
# Datasets
parser.add_argument('--root', type=str, default='/media/sdb1/zzj/datasets')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--train_sample_mode', type=str, default='random_crop',
                    help="random_crop, rrs")
parser.add_argument('--test_sample_mode', type=str, default='begin_crop',
                    help="begin_crop, rrs0")
# Augment
parser.add_argument('--seq_len', type=int, default=4,
                    help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=8,
                    help="stride of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max_epoch', default=240, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=32, type=int)
parser.add_argument('--lr', default=0.0003, type=float)
parser.add_argument('--stepsize', default=[40, 80, 120, 160, 200], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float)
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--distance', type=str, default='cosine',
                    help="euclidean or cosine")
parser.add_argument('--num_instances', type=int, default=4,
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='me')
# Miscs
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--eval_step', type=int, default=10)
parser.add_argument('--start_eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log-mars-ap3d')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu', default='0, 1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def fix_seed(seed):
    print("==========fix seed : {}==========".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False


def main():
    fix_seed(626)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))

    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    # Data augmentation
    spatial_transform_train = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.LGT(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    temporal_transform_train = TT.TemporalSample(
        mode=args.train_sample_mode, seq_len=args.seq_len)

    spatial_transform_test = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalSample(
        mode=args.test_sample_mode, seq_len=args.seq_len)

    pin_memory = True if use_gpu else False

    if args.dataset not in ['mars', 'LSVID']:
        trainloader = DataLoader(
            VideoDataset(dataset.train_dense, spatial_transform=spatial_transform_train,
                         temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler(
                dataset.train_dense, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True)
    else:
        trainloader = DataLoader(
            VideoDataset(dataset.train, spatial_transform=spatial_transform_train,
                         temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler(
                dataset.train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True)

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(
        name=args.arch, num_classes=dataset.num_train_pids)
    print("Model size: {:.5f}M".format(sum(p.numel()
          for p in model.parameters())/1000000.0))

    if 'multiloss' in args.arch:
        criterion_xent1 = nn.CrossEntropyLoss()
        criterion_htri1 = TripletLoss(
            margin=args.margin, distance=args.distance)
        criterion_xent2 = nn.CrossEntropyLoss()
        criterion_htri2 = TripletLoss(
            margin=args.margin, distance=args.distance)
        criterion_xent3 = nn.CrossEntropyLoss()
        criterion_htri3 = TripletLoss(
            margin=args.margin, distance=args.distance)
        criterion_xent4 = nn.CrossEntropyLoss()
        criterion_htri4 = TripletLoss(
            margin=args.margin, distance=args.distance)
        criterion_xent = [criterion_xent1, criterion_xent2,
                          criterion_xent3, criterion_xent4]
        criterion_htri = [criterion_htri1, criterion_htri2,
                          criterion_htri3, criterion_htri4]
    else:
        criterion_xent = nn.CrossEntropyLoss()
        criterion_htri = TripletLoss(
            margin=args.margin, distance=args.distance)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        # start_epoch = checkpoint['epoch']

    model = model.cuda()
    if use_gpu:
        model = nn.DataParallel(model)
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):

        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri,
              optimizer, trainloader, use_gpu)
        scheduler.step()
        train_time += round(time.time() - start_train_time)

        if (epoch+1) >= args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            with torch.no_grad():
                # test using 4 frames
                rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            checkpoint = {'model': state_dict,
                          'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, is_best, osp.join(
                args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print(
        "==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(
        elapsed, train_time))


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):

    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (vids, pids, camids, img_paths) in enumerate(trainloader):
        if (pids-pids[0]).sum() == 0:
            # can't compute triplet loss
            continue

        if use_gpu:
            vids, pids, camids = vids.cuda(), pids.cuda(), camids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward

        outputs, features = model(vids)

        # combine hard triplet loss with cross entropy loss
        if type(criterion_xent) == list:

            for i in range(len(criterion_xent)):
                if i == 0:
                    xent_loss = criterion_xent[i](outputs[i], pids)
                    htri_loss = criterion_htri[i](features[i], pids)
                    loss = criterion_xent[i](
                        outputs[i], pids)+criterion_htri[i](features[i], pids)
                else:
                    xent_loss += criterion_xent[i](outputs[i], pids)
                    htri_loss += criterion_htri[i](features[i], pids)
                    loss = loss + \
                        criterion_xent[i](outputs[i], pids) + \
                        criterion_htri[i](features[i], pids)
        else:
            htri_loss = criterion_htri(features, pids)
            xent_loss = criterion_xent(outputs, pids)
            loss = htri_loss+xent_loss
        loss.backward()
        optimizer.step()

        # statistics
        try:
            _, preds = torch.max(outputs[3].data, 1)
        except:
            _, preds = torch.max(outputs.data, 1)
        batch_corrects.update(
            torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '
          'Xent:{xent.avg:.4f} '
          'Htri:{htri.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
              epoch+1, batch_time=batch_time,
              data_time=data_time, loss=batch_loss,
              xent=batch_xent_loss, htri=batch_htri_loss,
              acc=batch_corrects))


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    # test using 4 frames
    since = time.time()
    model.eval()

    qf, q_pids, q_camids, q_paths, q_act = [], [], [], [], []
    for batch_idx, (vids, pids, camids, img_paths) in enumerate(queryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)

        feat = feat.mean(2)
        feat = model.module.bn(feat)
        feat = feat.data.cpu()

        qf.append(feat)
        q_pids.extend(pids)
        q_camids.extend(camids)
        q_paths.append(np.asarray(img_paths).transpose())

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_paths = np.concatenate(q_paths)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    gf, g_pids, g_camids, g_paths = [], [], [], []
    for batch_idx, (vids, pids, camids, img_paths) in enumerate(galleryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)
        feat = feat.mean(2)
        feat = model.module.bn(feat)
        feat = feat.data.cpu()

        gf.append(feat)
        g_pids.extend(pids)
        g_camids.extend(camids)
        g_paths.append(np.asarray(img_paths).transpose())

    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    g_paths = np.concatenate(g_paths)

    if args.dataset in ['mars', 'LSVID']:
        print(args.dataset)
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)
        g_paths = np.concatenate([q_paths, g_paths])
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
            distmat[i] = - torch.mm(qf[i:i+1], gf.t()) 
    # distmat
    distmat = distmat.numpy()
    print("Ori Computing CMC and mAP")
    cmc, mAP = evaluate_ori(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(
        cmc[0], cmc[4], cmc[9], mAP))
    print("------------------")

    # print("Vis Computing CMC and mAP")
    # cmc, mAP = evaluate_vis(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths,plot_ranking=True)

    # print("Results ----------")
    # print("Results ----------")
    # print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    # print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()
