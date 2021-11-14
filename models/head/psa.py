"""Point-wise Spatial Attention Network"""
import torch
import torch.nn as nn


up_kwargs = {'mode': 'bilinear', 'align_corners': True}
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


class PSAHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=6, norm_layer=norm_layer, in_index=3):
        super(PSAHead, self).__init__()
        self.in_index = in_index
        # psa_out_channels = crop_size // stride_rate ** 2
        psa_out_channels = (512 // 32) ** 2
        self.psa = _PointwiseSpatialAttention(in_channels, psa_out_channels, norm_layer)

        self.conv_post = _ConvBNReLU(psa_out_channels, in_channels, 1, norm_layer=norm_layer)
        self.project = nn.Sequential(
            _ConvBNReLU(in_channels * 2, in_channels // 2, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(in_channels // 2, num_classes, 1))

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        global_feature = self.psa(x)
        out = self.conv_post(global_feature)
        out = torch.cat([x, out], dim=1)
        out = self.project(out)

        return out


class _PointwiseSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_PointwiseSpatialAttention, self).__init__()
        reduced_channels = out_channels // 2
        self.collect_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)
        self.distribute_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)

    def forward(self, x):
        collect_fm = self.collect_attention(x)
        distribute_fm = self.distribute_attention(x)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        return psa_fm


class _AttentionGeneration(nn.Module):
    def __init__(self, in_channels, reduced_channels, out_channels, norm_layer):
        super(_AttentionGeneration, self).__init__()
        self.conv_reduce = _ConvBNReLU(in_channels, reduced_channels, 1, norm_layer=norm_layer)
        self.attention = nn.Sequential(
            _ConvBNReLU(reduced_channels, reduced_channels, 1, norm_layer=norm_layer),
            nn.Conv2d(reduced_channels, out_channels, 1, bias=False))

        self.reduced_channels = reduced_channels

    def forward(self, x):
        reduce_x = self.conv_reduce(x)
        attention = self.attention(reduce_x)
        n, c, h, w = attention.size()
        attention = attention.view(n, c, -1)
        reduce_x = reduce_x.view(n, self.reduced_channels, -1)
        fm = torch.bmm(reduce_x, torch.softmax(attention, dim=1))
        fm = fm.view(n, self.reduced_channels, h, w)

        return fm
