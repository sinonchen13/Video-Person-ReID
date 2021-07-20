from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
from .gem_pooling import GeneralizedMeanPoolingP

class MEModule(nn.Module):
    def __init__(self, channel, reduction=32, n_segment=4):
        print("MEModule")
        super(MEModule, self).__init__()

        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # Motion Module
        self.Mconv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.Mbn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.Mconv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        # all share
        self.Mconv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.Mbn3 = nn.BatchNorm2d(num_features=self.channel)

    def forward(self, x):

        Mbottleneck = self.Mconv1(x)
        Mbottleneck = self.Mbn1(Mbottleneck)

        Mreshape_bottleneck = Mbottleneck.view(
            (-1, self.n_segment) + Mbottleneck.size()[1:])  # b,t,c,h,w
        Mconv_bottleneck = self.Mconv2(Mbottleneck)  # 相减前的卷积
        Mreshape_conv_bottleneck = Mconv_bottleneck.view(
            (-1, self.n_segment) + Mconv_bottleneck.size()[1:])  # b,t,c,h,w

        Mt_fea, __ = Mreshape_bottleneck.split([self.n_segment-1, 1], dim=1)
        __, MtPlusone_fea = Mreshape_conv_bottleneck.split(
            [1, self.n_segment-1], dim=1)

        diff_fea = MtPlusone_fea - Mt_fea
        diff_fea_pluszero = F.pad(
            diff_fea, (0, 0, 0, 0, 0, 0, 0, 1), mode="constant", value=0)
        diff_fea_pluszero = diff_fea_pluszero.view(
            (-1,) + diff_fea_pluszero.size()[2:])  # bt,c,h,w

        my = self.avg_pool(diff_fea_pluszero)  # bt,c
        my = self.Mconv3(my)
        my = self.Mbn3(my)
        my = self.sigmoid(my)
        my = my - 0.5
        output = x + x * my.expand_as(x)
        return output


class MeGemPModule(nn.Module):
    def __init__(self, channel, reduction=32, n_segment=4):
        print("MeGemPModule")
        super(MeGemPModule, self).__init__()

        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.avg_pool = GeneralizedMeanPoolingP(mode="2d")
        self.sigmoid = nn.Sigmoid()

        # Motion Module
        self.Mconv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.Mbn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.Mconv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        # all share
        self.Mconv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.Mbn3 = nn.BatchNorm2d(num_features=self.channel)

    def forward(self, x):

        Mbottleneck = self.Mconv1(x)
        Mbottleneck = self.Mbn1(Mbottleneck)

        Mreshape_bottleneck = Mbottleneck.view(
            (-1, self.n_segment) + Mbottleneck.size()[1:])  # b,t,c,h,w
        Mconv_bottleneck = self.Mconv2(Mbottleneck)  # 相减前的卷积
        Mreshape_conv_bottleneck = Mconv_bottleneck.view(
            (-1, self.n_segment) + Mconv_bottleneck.size()[1:])  # b,t,c,h,w

        Mt_fea, __ = Mreshape_bottleneck.split([self.n_segment-1, 1], dim=1)
        __, MtPlusone_fea = Mreshape_conv_bottleneck.split(
            [1, self.n_segment-1], dim=1)

        diff_fea = MtPlusone_fea - Mt_fea
        diff_fea_pluszero = F.pad(
            diff_fea, (0, 0, 0, 0, 0, 0, 0, 1), mode="constant", value=0)
        diff_fea_pluszero = diff_fea_pluszero.view(
            (-1,) + diff_fea_pluszero.size()[2:])  # bt,c,h,w

        my = self.avg_pool(diff_fea_pluszero)  # bt,c
        my = self.Mconv3(my)
        my = self.Mbn3(my)
        my = self.sigmoid(my)
        my = my - 0.5
        output = x + x * my.expand_as(x)
        return output




if __name__ == "__main__":
    # pass
    net = MEModule(2048)
    x = net(torch.rand(8*4, 2048, 16, 8))
    print(x.shape)
