import torch.nn as nn
try:
    from .resnet import resnet50_v1b
except:
    from resnet import resnet50_v1b
import torch.nn.functional as F


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, backbone='resnet50', dilated=True, pretrained_base=False, **kwargs):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class SegBase(SegBaseModel):

    def __init__(self, nclass, backbone='resnet50', pretrained_base=False, **kwargs):
        super(SegBase, self).__init__(nclass, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _FCNHead(2048, nclass, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    from tools.flops_params_fps_count import flops_params_fps
    model = SegBase(nclass=6)
    flops_params_fps(model)










