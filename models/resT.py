# paper: ResT: An Efficient Transformer for Visual Recognition
# code: https://github.com/wofmanaf/ResT
import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.head import *
import torch.nn.functional as F

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

    # load state_dict
    load_state_dict(model, state_dict, strict)
    print('load pretrained weight strct={}'.format(strict))
    return checkpoint


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class GL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size+1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.max_pool(x)

        if self.with_pos:
            x = self.pos(x)
        return x


class ResTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm, apply_transform=False):
        super().__init__()
        self.depths = depths
        self.apply_transform = apply_transform

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dim[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dim[0], out_ch=embed_dim[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dim[1], out_ch=embed_dim[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dim[2], out_ch=embed_dim[3], with_pos=True)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dim[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dim[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dim[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dim[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])

        self.norm = norm_layer(embed_dim[3])

    def init_weights(self, pretrained=None, strict=False):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(self, pretrained, strict=strict)
            print('load pretained weight strict={}'.format(strict))
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        outs.append(x)
        # stage 2
        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        outs.append(x)
        # stage 3
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        outs.append(x)
        # stage 4
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)

        x = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        outs.append(x)

        return tuple(outs)


class ResT(nn.Module):

    def __init__(self, nclass, embed_dim, depths, num_heads, mlp_ratios, sr_ratios, apply_transform, qkv_bias,
                 aux=False, pretrained_root=None, head='seghead', edge_aux=False):
        super(ResT, self).__init__()
        self.aux = aux
        self.edge_aux = edge_aux
        self.head_name = head
        self.backbone = ResTransformer(embed_dim=embed_dim,
                                       depths=depths,
                                       num_heads=num_heads,
                                       drop_path_rate=0.3,
                                       mlp_ratios=mlp_ratios,
                                       sr_ratios=sr_ratios,
                                       apply_transform=apply_transform,
                                       qkv_bias=qkv_bias,
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


def rest_tiny(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/rest_lite.pth'
    else:
        pretrained_root = None
    model = ResT(
        nclass=nclass, aux=aux, head=head, edge_aux=edge_aux,
        embed_dim=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, pretrained_root=pretrained_root)
    return model


def rest_small(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/rest_small.pth'
    else:
        pretrained_root = None
    model = ResT(
        nclass=nclass, aux=aux, head=head, edge_aux=edge_aux,
        embed_dim=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        depths=[2, 2, 6, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, pretrained_root=pretrained_root)
    return model


def rest_base(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/rest_base.pth'
    else:
        pretrained_root = None
    model = ResT(
        nclass=nclass, aux=aux, head=head, edge_aux=edge_aux,
        embed_dim=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        depths=[2, 2, 6, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, pretrained_root=pretrained_root)
    return model


def rest_large(nclass, pretrained=False, aux=False, head='uperhead', edge_aux=False):
    if pretrained:
        pretrained_root = './pretrained_weights/rest_large.pth'
    else:
        pretrained_root = None
    model = ResT(
        nclass=nclass, aux=aux, head=head, edge_aux=edge_aux,
        embed_dim=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, pretrained_root=pretrained_root)
    return model


if __name__ == '__main__':
    """Notice if torch1.6, try to replace a / b with torch.true_divide(a, b)"""
    from tools.flops_params_fps_count import flops_params_fps

    model_base = rest_base(nclass=6, aux=True, edge_aux=False, head='uperhead', pretrained=False)

    flops_params_fps(model_base)

