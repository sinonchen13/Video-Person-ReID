from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import models
from module.ME_Modules import MEModule
from .ResNet import Bottleneck, weights_init_kaiming, weights_init_classifier
from module.coordatt import CoordAtt
from module.gem_pooling import GeneralizedMeanPoolingP
import random
from .ResNet import ResNet50

__all__ = ['ME_ResNet50',"MultiLoss_ResNet50","CoordAtt_ResNet50","Baseline" ]

#base 
class ResNet_Video_Insert(nn.Module):
    def __init__(self, insert_module,last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3], insert_layers=[0, 0, 0, 0]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        ins_idx = 0
        self.Ins_1 = nn.ModuleList([insert_module(self.inplanes)
                                  for i in range(insert_layers[ins_idx])])
        self.Ins_1_idx = sorted([layers[0]-(i+1)
                               for i in range(insert_layers[ins_idx])])
        ins_idx += 1

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.Ins_2 = nn.ModuleList([insert_module(self.inplanes)
                                  for i in range(insert_layers[ins_idx])])
        self.Ins_2_idx = sorted([layers[1]-(i+1)
                               for i in range(insert_layers[ins_idx])])
        ins_idx += 1

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.Ins_3 = nn.ModuleList([insert_module(self.inplanes)
                                  for i in range(insert_layers[ins_idx])])
        self.Ins_3_idx = sorted([layers[2]-(i+1)
                               for i in range(insert_layers[ins_idx])])
        ins_idx += 1

        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)
        self.Ins_4 = nn.ModuleList([insert_module(self.inplanes)
                                  for i in range(insert_layers[ins_idx])])
        self.Ins_4_idx = sorted([layers[3]-(i+1)
                               for i in range(insert_layers[ins_idx])])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.ModuleList(layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        # Layer 1
        Ins1_counter = 0
        if len(self.Ins_1_idx) == 0:
            self.Ins_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.Ins_1_idx[Ins1_counter]:
                x = self.Ins_1[Ins1_counter](x)
                Ins1_counter += 1
        # Layer 2
        Ins2_counter = 0
        if len(self.Ins_2_idx) == 0:
            self.Ins_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.Ins_2_idx[Ins2_counter]:
                x = self.Ins_2[Ins2_counter](x)
                Ins2_counter += 1

        # Layer 3
        Ins3_counter = 0
        if len(self.Ins_3_idx) == 0:
            self.Ins_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.Ins_3_idx[Ins3_counter]:
                x = self.Ins_3[Ins3_counter](x)
                Ins3_counter += 1

        # Layer 4
        Ins4_counter = 0
        if len(self.Ins_4_idx) == 0:
            self.Ins_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.Ins_4_idx[Ins4_counter]:
                x = self.Ins_4[Ins4_counter](x)
                Ins4_counter += 1

        return x


class ME_ResNet50(nn.Module):
    def __init__(self, num_classes):
        print("ME_ResNet50")
        super(ME_ResNet50, self).__init__()
        self.backbone = ResNet_Video_Insert(MEModule,insert_layers=[0,2,3,0])
        original = models.resnet50(pretrained=True).state_dict()
        for key in original:
            if key.find('fc') != -1:
                continue
            self.backbone.state_dict()[key].copy_(original[key])
        del original

        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, num_classes)
        self.classifier.apply(weights_init_classifier)
        self.gemp2d = GeneralizedMeanPoolingP(mode="2d")

        

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = self.backbone(x)
        x = self.gemp2d(x)  # bt,c,1,1
        x = x.view(b, t, -1)  # b,t,c
        x = x.permute(0, 2, 1)  # b,c,t
        
        if not self.training:
            return x
        x=x.mean(2)
        f = self.bn(x)
        y = self.classifier(f)
        return y, f


class MultiLoss_ResNet50(nn.Module):
    def __init__(self, num_classes):
        print("MultiLoss_ResNet50")
        super(MultiLoss_ResNet50, self).__init__()
        self.backbone = ResNet50()
        original = models.resnet50(pretrained=True).state_dict()
        for key in original:
            if key.find('fc') != -1:
                continue
            self.backbone.state_dict()[key].copy_(original[key])
        del original

        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)

        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier1.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(2048, num_classes)
        self.classifier2.apply(weights_init_classifier)

        self.classifier3 = nn.Linear(2048, num_classes)
        self.classifier3.apply(weights_init_classifier)

        self.classifier4 = nn.Linear(2048, num_classes)
        self.classifier4.apply(weights_init_classifier)

        self.classifier = [self.classifier1, self.classifier2,
                           self.classifier3, self.classifier4]

        self.gemp2d = GeneralizedMeanPoolingP(mode="2d")

        self.gemp1d = GeneralizedMeanPoolingP(mode="1d")
        

    def forward(self, x):
        multi_ffeat = []
        f_all = []
        y_all = []
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = self.backbone(x)
        x = self.gemp2d(x)  # bt,c,1,1
        x = x.view(b, t, -1)  # b,t,c
        x = x.permute(0, 2, 1)  # b,c,t
        
        for i in range(t):
            if not self.training:
                tmp = list(range(i+1))
            else:
                tmp = random.sample(list(range(t)), i+1)
            tmp_x = x[:, :, tmp]
            tmp_x = self.gemp1d(tmp_x)
            multi_ffeat.append(tmp_x)
        if not self.training:
            return multi_ffeat[3]
        for i in range(len(multi_ffeat)):
            multi_ffeat[i]=multi_ffeat[i].mean(2)
            tmp_f = self.bn(multi_ffeat[i])
            f_all.append(tmp_f)
            tmp_y = self.classifier[i](tmp_f)
            y_all.append(tmp_y)
        return y_all, f_all


class CoordAtt_ResNet50(nn.Module):
    def __init__(self, num_classes):
        print("CoordAtt_ResNet50")
        super(CoordAtt_ResNet50, self).__init__()
        self.backbone = ResNet_Video_Insert(CoordAtt,insert_layers=[0,0,0,1])
        original = models.resnet50(pretrained=True).state_dict()
        for key in original:
            if key.find('fc') != -1:
                continue
            self.backbone.state_dict()[key].copy_(original[key])
        del original

        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, num_classes)
        self.classifier.apply(weights_init_classifier)

        self.gemp2d = GeneralizedMeanPoolingP(mode="2d")

        

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = self.backbone(x)
        x = self.gemp2d(x)  # bt,c,1,1
        x = x.view(b, t, -1)  # b,t,c
        x = x.permute(0, 2, 1)  # b,c,t
        if not self.training:
            return x
        x = x.mean(2)
        f = self.bn(x)
        y = self.classifier(f)
        return y, f


class Baseline(nn.Module):
    def __init__(self, num_classes):
        print("Baseline")
        super(Baseline, self).__init__()
        self.backbone = ResNet50()
        original = models.resnet50(pretrained=True).state_dict()
        for key in original:
            if key.find('fc') != -1:
                continue
            self.backbone.state_dict()[key].copy_(original[key])
        del original

        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, num_classes)
        self.classifier.apply(weights_init_classifier)
        self.gemp2d = GeneralizedMeanPoolingP(mode="2d")

        

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = self.backbone(x)
        x = self.gemp2d(x)  
        x = x.view(b, t, -1)  
        x = x.permute(0, 2, 1) 
        
        if not self.training:
            return x
        x=x.mean(2)
        f = self.bn(x)
        y = self.classifier(f)
        return y, f


if __name__ == "__main__":

    net = MEModule(2048)
    x = net(torch.ones(8*4, 2048, 16, 8))
    print(x.shape)
