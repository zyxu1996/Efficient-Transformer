import torch.nn as nn
import torch
import torch.nn.functional as F


up_kwargs = {'mode': 'bilinear', 'align_corners': False}


class EdgeHead(nn.Module):
    """Edge awareness module"""

    def __init__(self, in_channels=[96, 192], channels=96, out_fea=2, in_index=[0, 1]):
        super(EdgeHead, self).__init__()
        self.in_index = in_index
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(True),
            nn.Conv2d(in_channels[0], channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels[1], in_channels[1], 1, 1, 0),
        #     nn.BatchNorm2d(in_channels[1]),
        #     nn.ReLU(True),
        #     nn.Conv2d(in_channels[1], channels, 1, 1, 0),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(True),
        # )
        self.conv3 = nn.Conv2d(channels, out_fea, 1, 1, 0)

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        x1, x2 = inputs
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        # edge2_fea = self.conv2(x2)

        edge1_fea = F.interpolate(edge1_fea, size=(h, w), **up_kwargs)
        # edge2_fea = F.interpolate(edge2_fea, size=(h, w), **up_kwargs)

        # edge_fea = torch.cat([edge1_fea, edge2_fea], dim=1)

        edge = self.conv3(edge1_fea)

        return edge
