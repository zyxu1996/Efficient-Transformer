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


class CEFPNHead(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], num_classes=6, channels=256,
                 norm_layer=norm_layer, up_kwargs=up_kwargs, in_index=[0, 1, 2, 3]):
        super(CEFPNHead, self).__init__()
        assert up_kwargs is not None
        self._up_kwargs = up_kwargs
        self.in_index = in_index
        self.C5_2_F4 = nn.Sequential(
                nn.Conv2d(in_channels[3], in_channels[2], kernel_size=1, bias=False),
                norm_layer(in_channels[2]),
                nn.ReLU(inplace=True))
        self.C4_2_F4 = nn.Sequential(
                nn.Conv2d(in_channels[2], channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True))
        self.C3_2_F3 = nn.Sequential(
                nn.Conv2d(in_channels[1], channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True))
        self.C2_2_F2 = nn.Sequential(
                nn.Conv2d(in_channels[0], channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True))

        fpn_out = []
        for _ in range(len(in_channels)):
            fpn_out.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True),
            ))
        self.fpn_out = nn.ModuleList(fpn_out)
        inter_channels = len(in_channels) * channels
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(512, num_classes, 1))
        # channel_attention_guide
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channels // 16, out_features=channels))
        self.sigmoid = nn.Sigmoid()

        # sub_pixel_context_enhancement
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels[-1], in_channels[-1] // 2, kernel_size=3, padding=1, bias=False),
                                    norm_layer(in_channels[-1] // 2),
                                    nn.ReLU())
        self.max_pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels[-1], in_channels[-1] * 2, 1, bias=False),
                                    norm_layer(in_channels[-1] * 2),
                                    nn.ReLU())
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels[-1], in_channels[-1] // 8, 1, bias=False),
                                    norm_layer(in_channels[-1] // 8),
                                    nn.ReLU())
        # inchannels to channels
        self.smooth1 = nn.Sequential(
                nn.Conv2d(in_channels[0], channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(
            nn.Conv2d(in_channels[0], channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True))
        self.smooth3 = nn.Sequential(
            nn.Conv2d(in_channels[0], channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True))

    def sub_pixel_conv(self, inputs, up_factor=2):
        b, c, h, w = inputs.shape
        assert c % (up_factor * up_factor) == 0
        inputs = inputs.permute(0, 2, 3, 1)  # b h w c
        inputs = inputs.view(b, h, w, c // (up_factor * up_factor), up_factor, up_factor)
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        inputs = inputs.view(b, h * up_factor, w * up_factor, c // (up_factor * up_factor)).permute(0, 3, 1, 2)
        inputs = inputs.contiguous()
        return inputs

    def channel_attention_guide(self, inputs):
        avgout = self.shared_MLP(self.avg_pool(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3)
        weights = self.sigmoid(avgout + maxout)
        output = weights * inputs
        return output

    def sub_pixel_context_enhancement(self, inputs):
        h, w = inputs.size()[2:]
        input1 = self.sub_pixel_conv(self.conv1(inputs))
        input2 = self.sub_pixel_conv(self.conv2(self.max_pool2(inputs)), up_factor=4)
        input3 = upsample(self.conv3(inputs), (h * 2, w * 2), **self._up_kwargs)
        output = input1 + input2 + input3
        output = self.smooth3(output)
        return output

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c5 = inputs[-1]
        c1_size = inputs[0].size()[2:]
        if hasattr(self, 'extramodule'):
            c5 = self.extramodule(c5)

        feat = self.sub_pixel_context_enhancement(c5)
        feat_up = upsample(self.channel_attention_guide(self.fpn_out[3](feat)), c1_size, **self._up_kwargs)
        fpn_features = [feat_up]

        feat = self.smooth1(self.sub_pixel_conv(self.C5_2_F4(c5))) + self.C4_2_F4(inputs[2])
        feat_up = upsample(self.channel_attention_guide(self.fpn_out[2](feat)), c1_size, **self._up_kwargs)
        fpn_features.append(feat_up)

        feats = []
        feats.append(self.C2_2_F2(inputs[0]))
        feats.append(self.smooth2(self.sub_pixel_conv(inputs[2])) + self.C3_2_F3(inputs[1]))

        for i in reversed(range(len(inputs) - 2)):
            feat_i = feats[i]
            feat = upsample(feat, feat_i.size()[2:], **self._up_kwargs)
            feat = feat + feat_i
            feat_up = upsample(self.channel_attention_guide(self.fpn_out[i](feat)), c1_size, **self._up_kwargs)
            fpn_features.append(feat_up)
        fpn_features = torch.cat(fpn_features, 1)

        return self.conv5(fpn_features)
