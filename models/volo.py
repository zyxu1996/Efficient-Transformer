# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Vision OutLOoker (VOLO) implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
import math
import numpy as np
from models.head import *

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


def load_state_dict(module, state_dict, strict=False):
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle
    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print(err_msg)


def resize_pos_embed_4d(posemb, posemb_new):
    '''return new position embedding'''
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    gs_old = posemb.shape[1]  # 14
    gs_new = posemb_new.shape[1]  # 24
    posemb_grid = posemb
    posemb_grid = posemb_grid.permute(0, 3, 1,
                                      2)  # [1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1)  # [1, dim, 24, 24]->[1, 24, 24, dim]
    return posemb_grid


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    ):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    old_posemb = state_dict['pos_embed']
    if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
        new_posemb = resize_pos_embed_4d(old_posemb, model.pos_embed)
        state_dict['pos_embed'] = new_posemb

    # load state_dict
    load_state_dict(model, state_dict, strict)
    print('load pretrained weight strct={}'.format(strict))
    return checkpoint


class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(torch.true_divide(H, self.stride)), math.ceil(torch.true_divide(W, self.stride))
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Outlooker(nn.Module):
    """
    Implementation of outlooker layer: which includes outlook attention + MLP
    Outlooker is the first stage in our VOLO
    --dim: hidden dim
    --num_heads: number of heads
    --mlp_ratio: mlp ratio
    --kernel_size: kernel size in each window for outlook attention
    return: outlooker layer
    """
    def __init__(self, dim, kernel_size, padding, stride=1,
                 num_heads=1,mlp_ratio=3., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, qkv_bias=False,
                 qk_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size,
                                     padding=padding, stride=stride,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim,  num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    """
    Implementation of Transformer,
    Transformer is the second stage in our VOLO
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Module):
    """
    Class attention layer from CaiT, see details in CaiT
    Class attention is the post stage in our VOLO, which is optional.
    """
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim,
                            self.head_dim * self.num_heads * 2,
                            bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[
            1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape(
            B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):
    """
    Class attention block from CaiT, see details in CaiT
    We use two-layers class attention in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


def get_block(block_type, **kargs):
    """
    get block by name, specifically for class attention block in here
    """
    if block_type == 'ca':
        return ClassBlock(**kargs)


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1,
                 patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]

        self.stem_conv = stem_conv
        if stem_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride,
                          padding=3, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.proj = nn.Conv2d(hidden_dim,
                              embed_dim,
                              kernel_size=patch_size // stem_stride,
                              stride=patch_size // stem_stride)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Module):
    """
    Image to Patch Embedding, downsampling between stage1 and stage2
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def outlooker_blocks(block_fn, index, dim, layers, num_heads=1, kernel_size=3,
                     padding=1,stride=1, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                     attn_drop=0, drop_path_rate=0., **kwargs):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(dim, kernel_size=kernel_size, padding=padding,
                               stride=stride, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                               drop_path=block_dpr))

    blocks = nn.Sequential(*blocks)

    return blocks


def transformer_blocks(block_fn, index, dim, layers, num_heads, mlp_ratio=3.,
                       qkv_bias=False, qk_scale=None, attn_drop=0,
                       drop_path_rate=0., **kwargs):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            block_fn(dim, num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     attn_drop=attn_drop,
                     drop_path=block_dpr))

    blocks = nn.Sequential(*blocks)

    return blocks


class VOLO_backbone(nn.Module):
    """
    Vision Outlooker, the main class of our model
    --layers: [x,x,x,x], four blocks in two stages, the first block is outlooker, the
              other three are transformer, we set four blocks, which are easily
              applied to downstream tasks
    --img_size, --in_chans, --num_classes: these three are very easy to understand
    --patch_size: patch_size in outlook attention
    --stem_hidden_dim: hidden dim of patch embedding, d1-d4 is 64, d5 is 128
    --embed_dims, --num_heads: embedding dim, number of heads in each block
    --downsamples: flags to apply downsampling or not
    --outlook_attention: flags to apply outlook attention or not
    --mlp_ratios, --qkv_bias, --qk_scale, --drop_rate: easy to undertand
    --attn_drop_rate, --drop_path_rate, --norm_layer: easy to undertand
    --post_layers: post layers like two class attention layers using [ca, ca],
                  if yes, return_mean=False
    --return_mean: use mean of all feature tokens for classification, if yes, no class token
    --return_dense: use token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --mix_token: mixing tokens as token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --pooling_scale: pooling_scale=2 means we downsample 2x
    --out_kernel, --out_stride, --out_padding: kerner size,
                                               stride, and padding for outlook attention
    """
    def __init__(self, layers, img_size=512, in_chans=3, patch_size=8,
                 stem_hidden_dim=64, embed_dims=None, num_heads=None, downsamples=None,
                 outlook_attention=None, mlp_ratios=None, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 pooling_scale=2, out_kernel=3, out_stride=2, out_padding=1):

        super().__init__()
        self.patch_embed = PatchEmbed(stem_conv=True, stem_stride=2, patch_size=patch_size,
                                      in_chans=in_chans, hidden_dim=stem_hidden_dim,
                                      embed_dim=embed_dims[0])

        # inital positional encoding, we add positional encoding after outlooker blocks
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size // pooling_scale,
                        img_size // patch_size // pooling_scale,
                        embed_dims[-1]))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # set the main block in network
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                # stage 1
                stage = outlooker_blocks(Outlooker, i, embed_dims[i], layers,
                                         downsample=downsamples[i], num_heads=num_heads[i],
                                         kernel_size=out_kernel, stride=out_stride,
                                         padding=out_padding, mlp_ratio=mlp_ratios[i],
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
            else:
                # stage 2
                stage = transformer_blocks(Transformer, i, embed_dims[i], layers,
                                           num_heads[i], mlp_ratio=mlp_ratios[i],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop_path_rate=drop_path_rate,
                                           attn_drop=attn_drop_rate,
                                           norm_layer=norm_layer)
                network.append(stage)

            if downsamples[i]:
                # downsampling between two stages
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))

        self.network = nn.ModuleList(network)
        trunc_normal_(self.pos_embed, std=.02)

    def init_weights(self, pretrained=None, strict=False):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(self, pretrained, strict=strict)
            print('load pretained weight strict={}'.format(strict))
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_embeddings(self, x):
        # patch embedding
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        out = []
        for idx, block in enumerate(self.network):
            if idx == 2:  # add positional encoding after outlooker blocks
                x = x + self.pos_embed
                x = self.pos_drop(x)
            x = block(x)
            out.append(x.permute(0, 3, 1, 2).contiguous())
        return out

    def forward(self, x):
        # step1: patch embedding
        x = self.forward_embeddings(x)

        # step2: tokens learning in the two stages
        out = self.forward_tokens(x)

        return tuple(out)


class VOLO(nn.Module):

    def __init__(self, nclass, embed_dim, layers, num_heads, mlp_ratios, downsamples, outlook_attention,
                 aux=False, pretrained_root=None, head='seghead', edge_aux=False, stem_hidden_dim=None):
        super(VOLO, self).__init__()
        self.aux = aux
        self.edge_aux = edge_aux
        self.head_name = head
        self.backbone = VOLO_backbone(layers=layers,
                                      embed_dims=embed_dim,
                                      num_heads=num_heads,
                                      mlp_ratios=mlp_ratios,
                                      downsamples=downsamples,
                                      outlook_attention=outlook_attention,
                                      stem_hidden_dim=stem_hidden_dim or 64,
                                      img_size=512
                                      )

        if self.head_name == 'apchead':
            self.decode_head = APCHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'aspphead':
            self.decode_head = ASPPHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'asppplushead':
            self.decode_head = ASPPPlusHead(in_channels=embed_dim[3], num_classes=nclass, in_index=[0, 3])

        if self.head_name == 'dahead':
            self.decode_head = DAHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'dnlhead':
            self.decode_head = DNLHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'fcfpnhead':
            self.decode_head = FCFPNHead(in_channels=embed_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.head_name == 'cefpnhead':
            self.decode_head = CEFPNHead(in_channels=embed_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.head_name == 'fcnhead':
            self.decode_head = FCNHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'gchead':
            self.decode_head = GCHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3, channels=512)

        if self.head_name == 'psahead':
            self.decode_head = PSAHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'psphead':
            self.decode_head = PSPHead(in_channels=embed_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'seghead':
            self.decode_head = SegHead(in_channels=embed_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        if self.head_name == 'unethead':
            self.decode_head = UNetHead(in_channels=embed_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        if self.head_name == 'uperhead':
            self.decode_head = UPerHead(in_channels=embed_dim, num_classes=nclass)

        if self.head_name == 'annhead':
            self.decode_head = ANNHead(in_channels=embed_dim[2:], num_classes=nclass, in_index=[2, 3], channels=512)

        if self.head_name == 'mlphead':
            self.decode_head = MLPHead(in_channels=embed_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.aux:
            self.auxiliary_head = FCNHead(num_convs=1, in_channels=embed_dim[2], num_classes=nclass, in_index=2, channels=256)

        if self.edge_aux:
            self.edge_head = EdgeHead(in_channels=embed_dim[0:2], in_index=[0, 1], channels=embed_dim[0])

        if pretrained_root is None:
            self.backbone.init_weights()
        else:
            if 'upernet' in pretrained_root:
                load_checkpoint(self, filename=pretrained_root, strict=False)
            else:
                self.backbone.init_weights(pretrained=pretrained_root, strict=False)

    def forward(self, x):
        size = x.size()[2:]
        outputs = []

        out_backbone = self.backbone(x)
        x0 = self.decode_head(out_backbone)
        if isinstance(x0, (list, tuple)):
            for out in x0:
                out = F.interpolate(out, size, **up_kwargs)
                outputs.append(out)
        else:
            x0 = F.interpolate(x0, size, **up_kwargs)
            outputs.append(x0)

        if self.aux:
            x1 = self.auxiliary_head(out_backbone)
            x1 = F.interpolate(x1, size, **up_kwargs)
            outputs.append(x1)

        if self.edge_aux:
            edge = self.edge_head(out_backbone)
            edge = F.interpolate(edge, size, **up_kwargs)
            outputs.append(edge)

        return outputs


def volo_d1(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/d1_224_84.2.pth.tar'
    else:
        pretrained_root = None

    layers = [4, 4, 8, 2]  # num of layers in the four blocks
    embed_dims = [192, 384, 384, 384]
    num_heads = [6, 12, 12, 12]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False] # do downsampling after first block
    outlook_attention = [True, False, False, False ]
    # first block is outlooker (stage1), the other three are transformer (stage2)
    model = VOLO(layers=layers,
                 embed_dim=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 nclass=nclass, aux=aux, head=head, edge_aux=edge_aux, pretrained_root=pretrained_root
                 )
    return model


def volo_d2(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/rest_lite.pth'
    else:
        pretrained_root = None

    layers = [6, 4, 10, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    # first block is outlooker (stage1), the other three are transformer (stage2)
    model = VOLO(layers=layers,
                 embed_dim=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 nclass=nclass, aux=aux, head=head, edge_aux=edge_aux, pretrained_root=pretrained_root
                 )
    return model


def volo_d3(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/rest_lite.pth'
    else:
        pretrained_root = None

    layers = [8, 8, 16, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    # first block is outlooker (stage1), the other three are transformer (stage2)
    model = VOLO(layers=layers,
                 embed_dim=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 nclass=nclass, aux=aux, head=head, edge_aux=edge_aux, pretrained_root=pretrained_root
                 )
    return model


def volo_d4(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/rest_lite.pth'
    else:
        pretrained_root = None

    layers = [8, 8, 16, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    # first block is outlooker (stage1), the other three are transformer (stage2)
    model = VOLO(layers=layers,
                 embed_dim=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 nclass=nclass, aux=aux, head=head, edge_aux=edge_aux, pretrained_root=pretrained_root
                 )
    return model


def volo_d5(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/d5_512_87.07.pth.tar'
    else:
        pretrained_root = None

    layers = [12, 12, 20, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    # first block is outlooker (stage1), the other three are transformer (stage2)
    model = VOLO(layers=layers,
                 embed_dim=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 stem_hidden_dim=128,
                 nclass=nclass, aux=aux, head=head, edge_aux=edge_aux, pretrained_root=pretrained_root
                 )
    return model


if __name__ == '__main__':
    """Notice if torch1.6, try to replace a / b with torch.true_divide(a, b)"""
    from tools.flops_params_fps_count import flops_params_fps

    model_base = volo_d1(nclass=6, aux=True, edge_aux=True, head='mlphead', pretrained=True)

    flops_params_fps(model_base)

