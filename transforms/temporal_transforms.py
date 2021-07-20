from __future__ import absolute_import

import random
import math
import numpy as np


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = list(frame_indices)

        while len(out) < self.size:
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, padding=True, pad_method='loop'):
        self.size = size
        self.padding = padding
        self.pad_method = pad_method

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = list(frame_indices[begin_index:end_index])

        if self.padding == True:
            if self.pad_method == 'loop':
                while len(out) < self.size:
                    for index in out:
                        if len(out) >= self.size:
                            break
                        out.append(index)
            else:
                while len(out) < self.size:
                    for index in out:
                        if len(out) >= self.size:
                            break
                        out.append(index)
                out.sort()

        return out


class TemporalSample(object):
    def __init__(self, mode, seq_len, stride=8):
        self.mode = mode
        self.seq_len = seq_len
        self.stride = stride

    def __call__(self, img_paths):
        if self.mode == "random_crop":
            return self.random_crop(img_paths)
        elif self.mode == "rrs":
            return self.rrs(img_paths)
        elif self.mode == "rrs0":
            return self.rrs0(img_paths)
        elif self.mode == "begin_crop":
            return self.begin_crop(img_paths)
        elif self.mode == "test_all_sampled":
            return self.test_all_sampled(img_paths)
        elif self.mode == "test_all_continuous":
            return self.test_all_continuous(img_paths)

    # train

    def random_crop(self, img_paths):
        frame_indices = list(img_paths)
        if len(frame_indices) >= self.seq_len * self.stride:
            rand_end = len(frame_indices) - \
                (self.seq_len - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.seq_len - 1) * self.stride + 1
            out = frame_indices[begin_index:end_index:self.stride]
        elif len(frame_indices) >= self.seq_len:
            index = np.random.choice(
                len(frame_indices), size=self.seq_len, replace=False)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.seq_len)]
        else:
            index = np.random.choice(
                len(frame_indices), size=self.seq_len, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.seq_len)]
        return out

    # train / fast test ?
    def rrs(self, img_paths):
        img_paths = list(img_paths)
        num = len(img_paths)
        indices = np.arange(0, num).astype(np.int32)
        num_pads = 0 if num % self.seq_len == 0 else self.seq_len - num % self.seq_len
        indices = np.concatenate(
            [indices, np.ones(num_pads).astype(np.int32)*(num-1)])
        assert len(indices) % self.seq_len == 0

        indices_pool = np.split(indices, self.seq_len)
        sampled_indices = []
        for part in indices_pool:
            sampled_indices.append(np.random.choice(part, 1)[0])
        sampled_imgs = []
        for index in sampled_indices:
            sampled_imgs.append(img_paths[index])
        return sampled_imgs

    def rrs0(self, img_paths):
        img_paths = list(img_paths)
        num = len(img_paths)
        indices = np.arange(0, num).astype(np.int32)
        num_pads = 0 if num % self.seq_len == 0 else self.seq_len - num % self.seq_len
        indices = np.concatenate(
            [indices, np.ones(num_pads).astype(np.int32)*(num-1)])
        assert len(indices) % self.seq_len == 0

        indices_pool = np.split(indices, self.seq_len)
        sampled_indices = []
        for part in indices_pool:
            sampled_indices.append(part[0])
        sampled_imgs = []
        for index in sampled_indices:
            sampled_imgs.append(img_paths[index])
        return sampled_imgs

    # val (test_0) / fast test ?
    def begin_crop(self, img_paths):

        frame_indices = list(img_paths)
        size = self.seq_len
        if len(frame_indices) >= (size - 1) * 8 + 1:
            out = frame_indices[0: (size - 1) * 8 + 1: 8]
        elif len(frame_indices) >= (size - 1) * 4 + 1:
            out = frame_indices[0: (size - 1) * 4 + 1: 4]
        elif len(frame_indices) >= (size - 1) * 2 + 1:
            out = frame_indices[0: (size - 1) * 2 + 1: 2]
        elif len(frame_indices) >= size:
            out = frame_indices[0:size:1]
        else:
            out = frame_indices[0:size]
            while len(out) < size:
                for index in out:
                    if len(out) >= size:
                        break
                    out.append(index)
        return out

    # test
    def test_all_sampled(self, img_paths):
        img_paths = list(img_paths)
        num = len(img_paths)
        indices = np.arange(0, num).astype(np.int32)
        num_pads = 0 if num % self.seq_len == 0 else self.seq_len - num % self.seq_len
        indices = np.concatenate(
            [indices, np.ones(num_pads).astype(np.int32)*(num-1)])
        assert len(indices) % self.seq_len == 0

        indices_pool = np.split(indices, self.seq_len)
        sampled_indices = []
        sampled_indices = np.vstack(indices_pool).T.flatten()
        sampled_imgs = []
        for index in sampled_indices:
            sampled_imgs.append(img_paths[index])
        return sampled_imgs

    # test
    def test_all_continuous(self, img_paths):
        img_paths = list(img_paths)
        num = len(img_paths)
        indices = np.arange(0, num).astype(np.int32)
        num_pads = 0 if num % self.seq_len == 0 else self.seq_len - num % self.seq_len
        indices = np.concatenate(
            [indices, np.ones(num_pads).astype(np.int32)*(num-1)])
        assert len(indices) % self.seq_len == 0

        indices_pool = np.split(indices, self.seq_len)
        sampled_indices = []
        sampled_indices = np.vstack(indices_pool).flatten()
        sampled_imgs = []
        for index in sampled_indices:
            sampled_imgs.append(img_paths[index])
        return sampled_imgs
