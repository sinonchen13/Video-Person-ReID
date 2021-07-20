import os
import sys
import torch
import torch.nn.functional as F
from torch import nn


class GeneralizedMeanPooling(nn.Module):
    r"""Copy from here: https://github.com/JDAI-CV/fast-reid/blob/e269caf8ab/fastreid/layers/pooling.py
    Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6, mode="2d"):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps
        self.mode = mode

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        if self.mode == "1d":
            return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size).pow(1. / (self.p+self.eps))
        elif self.mode == "2d":
            return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / (self.p+self.eps))
        elif self.mode == "3d":
            return torch.nn.functional.adaptive_avg_pool3d(x, self.output_size).pow(1. / (self.p+self.eps))
        else:
            assert self.mode in ["1d","2d", "3d"]

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    r""" Same, but norm is trainable
    """

    def __init__(self, mode, norm=3.0, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(
            norm, output_size, eps, mode)
        self.p = nn.Parameter(torch.ones(1, dtype=torch.float32) * norm)
