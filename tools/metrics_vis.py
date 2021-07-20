from __future__ import print_function, absolute_import
import numpy as np
import os.path as osp
import errno
import os
import cv2
from PIL import Image
import math

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def add_border(im, border_width, value):
    """Add color border around an image. The resulting image size is not changed.
    Args:
      im: numpy array with shape [3, im_h, im_w]
      border_width: scalar, measured in pixel
      value: scalar, or numpy array with shape [3]; the color of the border
    Returns:
      im: numpy array with shape [3, im_h, im_w]
    """
    assert (im.ndim == 3) and (im.shape[0] == 3)
    im = np.copy(im)

    if isinstance(value, np.ndarray):
        # reshape to [3, 1, 1]
        value = value.flatten()[:, np.newaxis, np.newaxis]
    im[:, :border_width, :] = value
    im[:, -border_width:, :] = value
    im[:, :, :border_width] = value
    im[:, :, -border_width:] = value

    return im


def save_vid_rank_result(query, top_gallery, save_path,q_act=None,g_act=None):
    """Save a query and its rank list as an image.
    Args:
        query (1D array): query sequence paths
        top_gallery (2D array): top gallery sequence paths
        save_path:
    """
    assert len(query) % 2 == 0
    n_cols = len(query) // 2
    
    query_id = int(query[0].split('/')[-2])
    top10_ids = [int(p[0].split('/')[-2]) for p in top_gallery]

    q_images = [read_im(q) for q in query]
    #print("q_act")
    #print(q_act.shape)
    #print("len q_images")
    #print(len(q_images))  4
    if q_act !=None:
        for i in range(len(q_images)):
            tmp = 0.4*q_images[i]+0.6*q_act[i].transpose(2,0,1)
            tmp[tmp>255]=255
            tmp=tmp.astype(np.uint8)
            q_images[i]=tmp
    q_im = make_img_grid(q_images, space=4, n_cols=n_cols, pad_val=255)
    images = [q_im]
    for gallery_path, gallery_id,gact in zip(top_gallery, top10_ids,g_act):
        g_images = [read_im(g) for g in gallery_path]
        if gact.all()!=None:
            for i in range(len(g_images)):
                tmp = 0.4*g_images[i]+0.6*gact[i].transpose(2,0,1)
                tmp[tmp>255]=255
                tmp=tmp.astype(np.uint8)
                g_images[i]=tmp

        g_im = make_img_grid(g_images, space=4, n_cols=n_cols, pad_val=255)

        # Add green boundary to true positive, red to false positive
        color = np.array([0, 255, 0]) if query_id == gallery_id else np.array([255, 0, 0])
        g_im = add_border(g_im, 3, color)
        images.append(g_im)

    im = make_QGimg_list(images, space=4, pad_val=255)
    im = im.transpose(1, 2, 0)
    Image.fromarray(im).save(save_path)


def make_img_grid(ims, space, n_cols, pad_val):
    """Make a grid of images with space in between.
    Args:
      ims: a list of [3, im_h, im_w] images
      space: the num of pixels between two images
      pad_val: scalar, or numpy array with shape [3]; the color of the space
    Returns:
      ret_im: a numpy array with shape [3, H, W]
    """
    n_rows = math.ceil(len(ims) / n_cols)
    assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
    h, w = ims[0].shape[1:]
    H = h * n_rows + space * (n_rows - 1)
    W = w * n_cols + space * (n_cols - 1)
    if isinstance(pad_val, np.ndarray):
        # reshape to [3, 1, 1]
        pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
    ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)

    for idx in range(len(ims)):
        curr_row = idx // n_cols
        curr_col = idx % n_cols
        start_h = curr_row * (h + space)
        start_w = curr_col * (w + space)
        ret_im[:, start_h:(start_h + h), start_w:(start_w + w)] = ims[idx]
    return ret_im


def make_QGimg_list(ims, space, pad_val):
    """Make a grid of images with space in between.
    Args:
      ims: a list of [3, im_h, im_w] images
      space: the num of pixels between two images
      pad_val: scalar, or numpy array with shape [3]; the color of the space
    Returns:
      ret_im: a numpy array with shape [3, H, W]
    """
    n_cols = len(ims)
    k_space = 5  # k_space means q_g space
    assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
    h, w = ims[0].shape[1:]
    H = h
    W = w * n_cols + space * (n_cols - 2) + k_space * space
    if isinstance(pad_val, np.ndarray):
        # reshape to [3, 1, 1]
        pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
    ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)

    ret_im[:, 0:h, 0:w] = ims[0]  # query image

    start_w = w + k_space * space
    for im in ims[1:]:
        end_w = start_w + w
        ret_im[:, 0:h, start_w:end_w] = im
        start_w = end_w + space
    return ret_im


def read_im(im_path):
    # shape [H, W, 3]
    im = np.asarray(Image.open(im_path))
    # Resize to (im_h, im_w) = (128, 64)
    resize_h_w = (128, 64)
    if (im.shape[0], im.shape[1]) != resize_h_w:
        im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    # shape [3, H, W]
    im = im.transpose(2, 0, 1)
    return im



def evaluate(distmat,
             q_pids, g_pids,
             q_camids, g_camids,
             q_paths=None, g_paths=None,q_act=None,g_act=None,
             plot_ranking=False,
             max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    if plot_ranking:
        rank_result_dir = '/media/sdb1/zzj/cache/res_vis/mars/'
        mkdir(rank_result_dir)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        
        # ---------------------- plot ranking results ------------------------
        if plot_ranking:
            #print("====g_act====")
            #print(len(g_act))
            #print(len(g_act[0]))
            #print(g_act[0][0].shape)
            assert q_paths is not None and g_paths is not None
            g_paths = np.asarray(g_paths)
            #print(g_paths.shape)
            #print(g_paths[0])
            top10 = g_paths[indices[q_idx]][keep][:10]
            #print(top10)
            top10_ids = g_pids[indices[q_idx]][keep][:10].tolist()

            if top10_ids[0] != q_pids[q_idx]:  # only plot ranking list of error top1
                save_vid_rank_result(q_paths[q_idx], top10,
                                          save_path=osp.join(rank_result_dir, osp.basename(q_paths[q_idx][0])),q_act=q_act[q_idx],g_act=g_act[indices[q_idx]][keep][:10])

                # save ground truth ranking list
                # TODO(NOTE): same id and different camera
                ground_truth = ((g_pids[indices[q_idx]] == q_pids[q_idx]) &
                                (g_camids[indices[q_idx]] != q_camids[q_idx]))
                ground_truth = np.where(ground_truth == 1)[0]
                top10 = g_paths[indices[q_idx]][ground_truth][:10]
                save_vid_rank_result(q_paths[q_idx], top10,
                                          save_path=osp.join(rank_result_dir,
                                                             osp.basename(q_paths[q_idx][0]).split('.')[
                                                                 0] + '_gt.jpg'),q_act=q_act[q_idx],g_act=g_act[indices[q_idx]][keep][:10])
        # ---------------------------------------------------------------------

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
