# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init

import mmcv
import os
from mmedit.core import tensor2img
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  nearest_stack_flow_warp, flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from .swin_irfm import SwinIRFM
from .pewarp_w2 import ImplicitWarpModule
from einops.layers.torch import Rearrange
from .op.deform_attn import deform_attn, DeformAttnPack

@BACKBONES.register_module()
class SintelSwinVSRNetFeatureAlignment(nn.Module):
    """VSR network structure for video super-resolution. (ablation study)

    Support only x4 upsampling.

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 embed_dim=60,
                 depths=[3, 3, 3],
                 num_heads=[6, 6, 6],
                 window_size=[2, 8, 8],
                 num_frames=2,
                 img_size = 64,
                 patch_size=1,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 alignment = 'of_warp',
                 interpolation = 'bicubic',
                 use_checkpoint=[False, False, False]):
        super().__init__()
        

        self.mid_channels = mid_channels
        self.embed_dim = embed_dim
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.conv_before_upsample = nn.Conv2d(embed_dim, mid_channels, 3, 1, 1)
        self.img_size = img_size
        self.patch_size = patch_size

        # feature extraction module
        if is_low_res_input:
            #self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
            self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # propagation branches
        self.swin_backbone_1 = SwinIRFM(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=[1, 8, 8],
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=use_checkpoint,
            upscale=4,
            img_range=1.,
            upsampler='pixelshuffle',
            resi_connection='1conv',
            num_frames=1)

        self.swin_backbone_2 = SwinIRFM(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=[2, 8, 8],
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=use_checkpoint,
            upscale=4,
            img_range=1.,
            upsampler='pixelshuffle',
            resi_connection='1conv',
            num_frames=2)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.alignment = alignment
        self.interpolation = interpolation
        print(self.alignment, self.interpolation)
        if self.alignment is 'nearest4':
            self.aggregation = nn.Conv2d(240, 60, 1, 1)
        elif self.alignment is 'ia':
            self.implicit_warp = ImplicitWarpModule(
                dim=embed_dim,
                pe_wrp=True,
                pe_x=True,
                pe_dim=embed_dim,
                num_heads=num_heads[0],
                pe_temp=0.01) 
        elif self.alignment is 'fgdc':
            self.deform_align = SecondOrderDeformableAlignment(
                embed_dim,
                embed_dim,
                3,
                padding=1,
                deform_groups=6,
                max_residue_magnitude=10)  
        elif self.alignment is 'fgda':
            self.deform_align = GuidedDeformAttnPack(embed_dim,
                                                    embed_dim,
                                                    attention_window=[3, 3],
                                                    attention_heads=12,
                                                    deformable_groups=12,
                                                    clip_size=1,
                                                    max_residue_magnitude=10)

    def forward(self, lqs, **kwargs):
        """Forward function.

        Args:
            lqs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flows_forward = kwargs['flows_forward']
        lqs_downsample = lqs.clone()
        n, t, c, h, w = lqs_downsample.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')


        feats = []
        for i in range(0, t):
            lr_curr = lqs_downsample[:, i, :, :, :]
            lr_curr = self.conv_first(lr_curr).unsqueeze(1)
            feats.append(self.swin_backbone_1(lr_curr))
        feats = torch.stack(feats, dim=1)

        # feature alignment
        forward_propagation = []
        for i in range(0, t):
            if i == t-1:
                forward_propagation.append(feats[:, i, :, :, :])
            else:
                if self.alignment is 'na':
                    forward_propagation.append(feats[:, i+1, :, :, :])
                elif self.alignment is 'of_warp':
                    forward_propagation.append(flow_warp(feats[:, i+1, :, :, :], flows_forward[:,i].permute(0,2,3,1), self.interpolation))
                elif self.alignment is 'nearest4':
                    stacked = nearest_stack_flow_warp(feats[:, i+1, :, :, :], flows_forward[:,i].permute(0,2,3,1), self.interpolation)
                    aggregated = self.aggregation(stacked)
                    forward_propagation.append(aggregated)
                elif self.alignment is 'pa':
                    forward_propagation.append(flow_warp_avg_patch(feats[:, i+1, :, :, :], flows_forward[:,i]))
                elif self.alignment is 'ia':
                    forward_propagation.append(self.implicit_warp(feats[:, i+1, :, :, :], feats[:, i, :, :, :], flows_forward[:,i].permute(0, 2, 3, 1)))
                elif self.alignment is 'fgdc':
                    feat_prop = feats[:, i+1, :, :, :]
                    flow_n1 = flows_forward[:,i]
                    feat_current = feats[:, i, :, :, :]
                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                    # flow-guided deformable convolution
                    cond = torch.cat([cond_n1, feat_current], dim=1)
                    forward_propagation.append(self.deform_align(feat_prop, cond, flow_n1))
                elif self.alignment is 'fgda':
                    feat_prop = feats[:, i+1, :, :, :]
                    flow_n1 = flows_forward[:,i]
                    feat_current = feats[:, i, :, :, :]
                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                    # flow-guided deformable convolution
                    forward_propagation.append(self.deform_align(feat_current.unsqueeze(1), feat_current.unsqueeze(1), feat_prop.unsqueeze(1), [cond_n1.unsqueeze(1)], [flow_n1.unsqueeze(1)]).squeeze(1))

                else:
                    raise NotImplementedError

        props = torch.stack(forward_propagation, dim=1)


        outputs = []
        for i in range(0, t):

            feat_curr = feats[:, i, :, :, :]
            feat_prop = props[:, i, :, :, :]
            feats_ = torch.stack([feat_curr, feat_prop], dim=1)
            #features
            hr = self.swin_backbone_2(feats_)
            hr = self.conv_before_upsample(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lqs[:, i, :, :, :])
            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

def flow_warp_avg_patch(x, flow, interpolation='nearest', padding_mode='zeros', align_corners=True):
    """Patch Alignment
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, 2,h, w). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    #todo microdoft transformer dtyle warping 

    # if x.size()[-2:] != flow.size()[1:3]:
    #     raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
    #                      f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # patch size is set to 8.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    flow = F.pad(flow, (0, pad_w, 0, pad_h), mode='reflect')
    hp = h + pad_h
    wp = w + pad_w
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, hp), torch.arange(0, wp))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    flow = F.avg_pool2d(flow, 8)
    flow = F.interpolate(flow, scale_factor=8, mode='nearest')
    flow = flow.permute(0, 2, 3, 1)
    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  #grid[:,:,:,0]是w
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x.float(), grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1):
        extra_feat = torch.cat([extra_feat, flow_1], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset_1 = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        
        
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset_1, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
    
class GuidedDeformAttnPack(DeformAttnPack):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(GuidedDeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv3d(self.in_channels * (1 + self.clip_size) + self.clip_size * 2, 64, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, self.clip_size * self.deformable_groups * self.attn_size * 2, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
        )
        self.init_offset()

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_k = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_v = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                  nn.Linear(self.proj_channels, self.in_channels),
                                  Rearrange('n d h w c -> n d c h w'))
        self.mlp = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                 Mlp(self.in_channels, self.in_channels * 2, self.in_channels),
                                 Rearrange('n d h w c -> n d c h w'))

    def init_offset(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, q, k, v, v_prop_warped, flows):
        offset = self.max_residue_magnitude * torch.tanh(
            self.conv_offset(torch.cat([q] + v_prop_warped + flows, 2).transpose(1, 2)).transpose(1, 2))
        offset = offset + flows[0].flip(2).repeat(1, 1, offset.size(2) // 2, 1, 1)
        b, t, c, h, w = offset.shape
        offset = offset.flatten(0, 1)

        q = self.proj_q(q).view(b * t, 1, self.proj_channels, h, w)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        #AssertionError: (torch.Size([4, 1, 120, 64, 64]), torch.Size([4, 1, 240, 64, 64]), torch.Size([4, 1, 216, 64, 64]))
        v = deform_attn(q, kv, offset, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups, self.clip_size).view(b, t, self.proj_channels, h,
                                                                                           w)
        v = self.proj(v)
        v = v + self.mlp(v)

        return v
        
class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
