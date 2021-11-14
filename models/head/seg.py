import torch
import torch.nn as nn
import torch.nn.functional as F

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv3x3 = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )
    return conv3x3


class SegHead(nn.Module):
    def __init__(self, in_channels=[96, 192, 384, 768], num_classes=6, in_index=[0, 1, 2, 3]):
        super(SegHead, self).__init__()
        self.in_index = in_index

        self.conv1 = conv3x3(in_channels[0], in_channels[0])
        self.conv2 = conv3x3(in_channels[1], in_channels[0])
        self.conv3 = conv3x3(in_channels[2], in_channels[0])
        self.conv4 = conv3x3(in_channels[3], in_channels[0])
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[0] * 4,
                out_channels=in_channels[0] * 4,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(in_channels[0] * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels[0] * 4,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        p2, p3, p4, p5 = inputs
        x2 = self.conv1(p2)
        x3 = F.interpolate(self.conv2(p3), scale_factor=2, **up_kwargs)
        x4 = F.interpolate(self.conv3(p4), scale_factor=4, **up_kwargs)
        x5 = F.interpolate(self.conv4(p5), scale_factor=8, **up_kwargs)
        x = torch.cat((x2, x3, x4, x5), dim=1)
        x = self.final_layer(x)

        return x
