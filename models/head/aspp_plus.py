import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp import ASPP_Module


up_kwargs = {'mode': 'bilinear', 'align_corners': False}
norm_layer = nn.BatchNorm2d


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=norm_layer):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPlusHead(nn.Module):
    def __init__(self, num_classes, in_channels, norm_layer=norm_layer, up_kwargs=up_kwargs, in_index=[0, 3]):
        super(ASPPPlusHead, self).__init__()
        self._up_kwargs = up_kwargs
        self.in_index = in_index
        self.aspp = ASPP_Module(in_channels, [12, 24, 36], norm_layer=norm_layer, up_kwargs=up_kwargs)
        self.c1_block = _ConvBNReLU(in_channels // 8, in_channels // 8, 3, padding=1, norm_layer=norm_layer)
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels // 4, in_channels // 4, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(in_channels // 4, in_channels // 4, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels // 4, num_classes, 1))

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c1, x = inputs
        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.interpolate(x, size, **self._up_kwargs)
        return self.block(torch.cat([x, c1], dim=1))
