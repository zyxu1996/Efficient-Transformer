import torch.nn as nn
import torch
from .base_decoder import BaseDecodeHead, resize

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768, norm_act=True):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm_act = norm_act
        if self.norm_act:
            self.norm = nn.LayerNorm(input_dim)
            self.act = nn.GELU()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm_act:
            x = self.norm(x)
        x = self.proj(x)
        if self.norm_act:
            x = self.act(x)
        return x


class MLPHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, in_channels=[96, 192, 384, 768], channels=512, num_classes=6, in_index=[0, 1, 2, 3]):
        super(MLPHead, self).__init__(input_transform='multiple_select', in_index=in_index,
                                            in_channels=in_channels, num_classes=num_classes, channels=channels)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=channels)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=channels)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=channels)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=channels)

        self.linear_c3_out = MLP(input_dim=channels, embed_dim=channels)
        self.linear_c2_out = MLP(input_dim=channels, embed_dim=channels)
        self.linear_c1_out = MLP(input_dim=channels, embed_dim=channels)

        self.linear_fuse = MLP(input_dim=channels * 4, embed_dim=channels)
        self.linear_pred = MLP(input_dim=channels, embed_dim=num_classes, norm_act=False)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        out = []
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c3.size()[2:], **up_kwargs)

        out.append(resize(_c4, size=c1.size()[2:], **up_kwargs))

        _c3 = self.linear_c3(c3).permute(0, 2, 1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = _c4 + _c3

        _c3_out = self.linear_c3_out(_c3).permute(0, 2, 1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        out.append(resize(_c3_out, size=c1.size()[2:], **up_kwargs))

        _c2 = self.linear_c2(c2).permute(0, 2, 1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c3 = resize(_c3, size=c2.size()[2:], **up_kwargs)
        _c2 = _c3 + _c2

        _c2_out = self.linear_c2_out(_c2).permute(0, 2, 1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        out.append(resize(_c2_out, size=c1.size()[2:], **up_kwargs))

        _c1 = self.linear_c1(c1).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], **up_kwargs)
        _c1 = _c2 + _c1

        _c1_out = self.linear_c1_out(_c1).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        out.append(_c1_out)

        _c = self.linear_fuse(torch.cat(out, dim=1)).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = self.dropout(_c)
        x = self.linear_pred(_c).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        return x
