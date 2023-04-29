import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY
from .psrt_sliding_arch import SwinIRFM
from .implicit_alignment import ImplicitWarpModule

@ARCH_REGISTRY.register()
class IART(nn.Module):
    """IART: Implicit Alignment Restoration Transformer.
    
    Paper:
        An Implicit Alignment for Video Super-Resolution
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 embed_dim=120,
                 depths=(6, 6, 6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6, 6, 6),
                 window_size=(3, 8, 8),
                 num_frames=3,
                 img_size = 64,
                 patch_size=1,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 use_checkpoint = [True, True, True, True],
                 spynet_path=None):

        super().__init__()
        self.mid_channels = mid_channels
        self.embed_dim = embed_dim
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.conv_before_upsample = nn.Conv2d(embed_dim, mid_channels, 3, 1, 1)
        self.img_size = img_size
        self.patch_size = patch_size
        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module
        if is_low_res_input:
            #self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
            self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # propagation branches
        self.swin_backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.swin_backbone[module] = SwinIRFM(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_channels,
                    embed_dim=embed_dim,
                    depths=depths,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=2.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm,
                    ape=False,
                    patch_norm=True,
                    use_checkpoint=use_checkpoint[i],
                    upscale=4,
                    img_range=1.,
                    upsampler='pixelshuffle',
                    resi_connection='1conv',
                    num_frames=num_frames)


        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        self.implicit_warp = ImplicitWarpModule(
            dim=embed_dim,
            pe_dim=embed_dim,
            num_heads=num_heads[0],
            pe_temp=0.01)

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def frame_alignment(self, feat_prop, feat_current, flow_n1):
        # pe_align: frame alignment using implicit alignment


    
        return self.implicit_warp(feat_prop, feat_current, flow_n1.permute(0, 2, 3, 1))

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.embed_dim, h, w)
        for i, idx in enumerate(frame_idx):
            if module_name == 'backward_1':
                feat_current = feats['spatial'][mapping_idx[idx]]
            if module_name == 'forward_1':
                feat_current = feats['backward_1'][mapping_idx[idx]]
            if module_name == 'backward_2':
                feat_current = feats['forward_1'][mapping_idx[idx]]
            if module_name == 'forward_2':
                feat_current = feats['backward_2'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = self.frame_alignment(feat_prop, feat_current, flow_n1)

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))

                    cond_n2 = self.frame_alignment(feat_n2, feat_current, flow_n2)



                cond = torch.stack([cond_n1, feat_current, cond_n2], dim=1)
                # patch alignment
                feat_prop = self.swin_backbone[module_name](cond)

            if i == 0:
                cond = torch.stack([feat_current, feat_current, feat_current], dim=1)
                feat_prop = self.swin_backbone[module_name](cond)

            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = feats['forward_2'][i]
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.conv_before_upsample(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for PSRT.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.conv_first(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.conv_first(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


    def flops(self):
        flops = 0
        h,w = self.img_size,self.img_size
        flops += h * w * 3 * self.embed_dim * 9
        for pipl_name,modules in self.swin_backbone.items():
            modules_flop = modules.flops()
            flops += modules_flop
            print(pipl_name, modules_flop / 1e9)
            print("\n")

        flops += h * w * self.embed_dim * self.mid_channels * 9
        flops += h * w * self.mid_channels * self.mid_channels* 4 * 9
        flops += 2*h * 2*w * self.mid_channels * self.mid_channels * 4 * 9
        flops += 4*h * 4*w * self.mid_channels * self.mid_channels  * 9
        flops += 4*h * 4*w * self.mid_channels * 3  * 9

        return flops



