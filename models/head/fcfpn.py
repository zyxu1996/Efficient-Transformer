###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division
import torch
import torch.nn as nn
from torch.nn.functional import upsample

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
norm_layer = nn.BatchNorm2d


class FCFPNHead(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], num_classes=6, channels=256,
                 norm_layer=norm_layer, up_kwargs=up_kwargs, in_index=[0, 1, 2, 3]):
        super(FCFPNHead, self).__init__()
        assert up_kwargs is not None
        self._up_kwargs = up_kwargs
        self.in_index = in_index
        fpn_lateral = []
        for inchannel in in_channels[:-1]:
            fpn_lateral.append(nn.Sequential(
                nn.Conv2d(inchannel, channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True),
            ))
        self.fpn_lateral = nn.ModuleList(fpn_lateral)
        fpn_out = []
        for _ in range(len(in_channels) - 1):
            fpn_out.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True),
            ))
        self.fpn_out = nn.ModuleList(fpn_out)
        self.c4conv = nn.Sequential(nn.Conv2d(in_channels[-1], channels, 3, padding=1, bias=False),
                                    norm_layer(channels),
                                    nn.ReLU())
        inter_channels = len(in_channels) * channels
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(512, num_classes, 1))

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c4 = inputs[-1]
        if hasattr(self, 'extramodule'):
            c4 = self.extramodule(c4)
        feat = self.c4conv(c4)
        c1_size = inputs[0].size()[2:]
        feat_up = upsample(feat, c1_size, **self._up_kwargs)
        fpn_features = [feat_up]

        for i in reversed(range(len(inputs) - 1)):
            feat_i = self.fpn_lateral[i](inputs[i])
            feat = upsample(feat, feat_i.size()[2:], **self._up_kwargs)
            feat = feat + feat_i
            feat_up = upsample(self.fpn_out[i](feat), c1_size, **self._up_kwargs)
            fpn_features.append(feat_up)
        fpn_features = torch.cat(fpn_features, 1)

        return self.conv5(fpn_features)
