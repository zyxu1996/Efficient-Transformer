import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init
from .base_decoder import BaseDecodeHead
import torch.nn.functional as F

norm_cfg = dict(type='BN', requires_grad=True)


class _SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.
    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.
    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(_SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context


class PPMConcat(nn.ModuleList):
    """Pyramid Pooling Module that only concat the features of each layer.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
    """

    def __init__(self, pool_scales=(1, 3, 6, 8)):
        super(PPMConcat, self).__init__(
            [nn.AdaptiveAvgPool2d(pool_scale) for pool_scale in pool_scales])

    def forward(self, feats):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(feats)
            ppm_outs.append(ppm_out.view(*feats.shape[:2], -1))
        concat_outs = torch.cat(ppm_outs, dim=2)
        return concat_outs


class SelfAttentionBlock(_SelfAttentionBlock):
    """Make a ANN used SelfAttentionBlock.
    Args:
        low_in_channels (int): Input channels of lower level feature,
            which is the key feature for self-attention.
        high_in_channels (int): Input channels of higher level feature,
            which is the query feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_scale (int): The scale of query feature map.
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, low_in_channels, high_in_channels, channels,
                 out_channels, share_key_query, query_scale, key_pool_scales,
                 conv_cfg, norm_cfg, act_cfg):
        key_psp = PPMConcat(key_pool_scales)
        if query_scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=query_scale)
        else:
            query_downsample = None
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=low_in_channels,
            query_in_channels=high_in_channels,
            channels=channels,
            out_channels=out_channels,
            share_key_query=share_key_query,
            query_downsample=query_downsample,
            key_downsample=key_psp,
            key_query_num_convs=1,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


class AFNB(nn.Module):
    """Asymmetric Fusion Non-local Block(AFNB)
    Args:
        low_in_channels (int): Input channels of lower level feature,
            which is the key feature for self-attention.
        high_in_channels (int): Input channels of higher level feature,
            which is the query feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
            and query projection.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, low_in_channels, high_in_channels, channels,
                 out_channels, query_scales, key_pool_scales, conv_cfg,
                 norm_cfg, act_cfg):
        super(AFNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(
                SelfAttentionBlock(
                    low_in_channels=low_in_channels,
                    high_in_channels=high_in_channels,
                    channels=channels,
                    out_channels=out_channels,
                    share_key_query=False,
                    query_scale=query_scale,
                    key_pool_scales=key_pool_scales,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.bottleneck = ConvModule(
            out_channels + high_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, low_feats, high_feats):
        """Forward function."""
        priors = [stage(high_feats, low_feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, high_feats], 1))
        return output


class APNB(nn.Module):
    """Asymmetric Pyramid Non-local Block (APNB)
    Args:
        in_channels (int): Input channels of key/query feature,
            which is the key feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, in_channels, channels, out_channels, query_scales,
                 key_pool_scales, conv_cfg, norm_cfg, act_cfg):
        super(APNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(
                SelfAttentionBlock(
                    low_in_channels=in_channels,
                    high_in_channels=in_channels,
                    channels=channels,
                    out_channels=out_channels,
                    share_key_query=True,
                    query_scale=query_scale,
                    key_pool_scales=key_pool_scales,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.bottleneck = ConvModule(
            2 * in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, feats):
        """Forward function."""
        priors = [stage(feats, feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, feats], 1))
        return output


class ANNHead(BaseDecodeHead):
    """Asymmetric Non-local Neural Networks for Semantic Segmentation.
    This head is the implementation of `ANNNet
    <https://arxiv.org/abs/1908.07678>`_.
    Args:
        project_channels (int): Projection channels for Nonlocal.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): The pooling scales of key feature map.
            Default: (1, 3, 6, 8).
    """

    def __init__(self,
                 project_channels=256,
                 query_scales=(1, ),
                 key_pool_scales=(1, 3, 6, 8),
                 in_channels=[384, 768],
                 num_classes=6,
                 in_index=[2, 3],
                 channels=512,
                 ):
        super(ANNHead, self).__init__(
            input_transform='multiple_select', in_channels=in_channels, num_classes=num_classes,
            in_index=in_index, channels=channels)
        assert len(self.in_channels) == 2
        low_in_channels, high_in_channels = self.in_channels
        self.project_channels = project_channels
        self.fusion = AFNB(
            low_in_channels=low_in_channels,
            high_in_channels=high_in_channels,
            out_channels=high_in_channels,
            channels=project_channels,
            query_scales=query_scales,
            key_pool_scales=key_pool_scales,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            high_in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.context = APNB(
            in_channels=self.channels,
            out_channels=self.channels,
            channels=project_channels,
            query_scales=query_scales,
            key_pool_scales=key_pool_scales,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        low_feats, high_feats = self._transform_inputs(inputs)
        output = self.fusion(low_feats, high_feats)
        output = self.dropout(output)
        output = self.bottleneck(output)
        output = self.context(output)
        output = self.cls_seg(output)

        return output