import math
import torch
import torch.nn as nn

class ImplicitWarpModule(nn.Module):
    """ Implicit Warp Module.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 pe_wrp=True,
                 pe_x=True,
                 pe_dim = 48,
                 pe_temp = 10000,
                 warp_padding='duplicate',
                 num_heads=8,
                 aux_loss_out = False,
                 aux_loss_dim = 3,
                 window_size=2,
                 qkv_bias=True,
                 qk_scale=None,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.dim = dim
        self.pe_wrp = pe_wrp
        self.pe_x = pe_x
        self.pe_dim = pe_dim
        self.pe_temp = pe_temp
        self.aux_loss_out = aux_loss_out

        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.window_size = (window_size, window_size)
        self.warp_padding = warp_padding
        self.q = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.k = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(pe_dim, dim, bias=qkv_bias)
        if self.aux_loss_out:
            self.proj = nn.Linear(dim, aux_loss_dim)

        self.softmax = nn.Softmax(dim=-1)
        
        self.register_buffer("position_bias", self.get_sine_position_encoding(self.window_size, pe_dim // 2, temperature=self.pe_temp, normalize=True))

        grid_h, grid_w = torch.meshgrid(
            torch.arange(0, self.window_size[0], dtype=int),
            torch.arange(0, self.window_size[1], dtype=int))

        self.num_values = self.window_size[0]*self.window_size[1]

        self.register_buffer("window_idx_offset", torch.stack((grid_h, grid_w), 2).reshape(self.num_values, 2))

    def gather_hw(self, x, idx1, idx2):
        # Linearize the last two dims and index in a contiguous x
        x = x.contiguous()
        lin_idx = idx2 + x.size(-1) * idx1
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        return x.gather(-1, lin_idx.unsqueeze(1).repeat(1,x.size(1),1))


    def forward(self, y, x, flow):
        # y: frame to be propagated.
        # x: frame propagated to.
        # flow: optical flow from x to y 
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                            f'flow ({flow.size()[1:3]}) are not the same.')
        n, c, h, w = x.size()
        # create mesh grid
        device = flow.device
        grid_h, grid_w = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
        grid = torch.stack((grid_h, grid_w), 2).repeat(n, 1, 1, 1)  # h, w, 2
        grid.requires_grad = False

        grid_wrp = grid + flow.flip(dims=(-1,)) # grid_wrp

        grid_wrp_flr = torch.floor(grid_wrp).int()
        grid_wrp_off = grid_wrp - grid_wrp_flr

        grid_wrp_flr = grid_wrp_flr.reshape(n, h*w, 2)
        grid_wrp_off = grid_wrp_off.reshape(n, h*w, 2)

        # get sliced windows
        grid_wrp = grid_wrp_flr.unsqueeze(2).repeat(1, 1, self.num_values, 1) + self.window_idx_offset # get 4x4 windows for each location.
        grid_wrp = grid_wrp.reshape(n, h*w*self.num_values, 2)
        if self.warp_padding == 'duplicate':
            idx0 = grid_wrp[:,:,0].clamp(min=0, max=h-1)
            idx1 = grid_wrp[:,:,1].clamp(min=0, max=w-1)
            wrp = self.gather_hw(y, idx0, idx1).reshape(n, c, h*w, self.num_values).permute(0,2,3,1).reshape(n, h*w*self.num_values, c)
        elif self.warp_padding == 'zero':
            invalid_h = torch.logical_or(grid_wrp[:,:,0]<0, grid_wrp[:,:,0]>h-1)
            invalid_w = torch.logical_or(grid_wrp[:,:,1]<0, grid_wrp[:,:,1]>h-1)
            invalid = torch.logical_or(invalid_h, invalid_w)

            idx0 = grid_wrp[:,:,0].clamp(min=0, max=h-1)
            idx1 = grid_wrp[:,:,1].clamp(min=0, max=w-1)

            wrp = self.gather_hw(y, idx0, idx1).reshape(n, c, h*w, self.num_values).permute(0,2,3,1).reshape(n, h*w*self.num_values, c)
            wrp[invalid] = 0
        else:
            raise ValueError(f'self.warp_padding: {self.warp_padding}')
        
        # add sin/cos positional encoding to 4x4 windows
        wrp_pe = self.position_bias.repeat(n, h*w, 1)

        if self.pe_wrp:
            wrp = wrp.repeat(1,1,self.pe_dim//c) + wrp_pe
        else:
            wrp = wrp.repeat(1,1,self.pe_dim//c)

        # add postional encoding to source pixel
        x = x.flatten(2).permute(0,2,1)
        x_pe = self.get_sine_position_encoding_points(grid_wrp_off, self.pe_dim // 2, temperature=self.pe_temp, normalize=True)

        if self.pe_x:
            x = x.repeat(1,1,self.pe_dim//c) + x_pe
        else:
            x = x.repeat(1,1,self.pe_dim//c)

        nhw = n*h*w
        
        kw = self.k(wrp).reshape(nhw, self.num_values, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) 
        vw = self.v(wrp).reshape(nhw, self.num_values, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        qx = self.q(x).reshape(nhw, self.num_heads, self.dim // self.num_heads).unsqueeze(1).permute(0, 2, 1, 3)


        attn = (qx * self.scale) @ kw.transpose(-2, -1)
        attn = self.softmax(attn)
        out = (attn @ vw).transpose(1, 2).reshape(nhw, 1, self.dim)

        out = out.squeeze(1)

        if self.aux_loss_out:
            out_rgb = self.proj(out).reshape(n, h, w, c).permute(0,3,1,2)
            return out.reshape(n, h, w, self.dim).permute(0,3,1,2), out_rgb
        else:
            return out.reshape(n, h, w, self.dim).permute(0,3,1,2)



    def get_sine_position_encoding_points(self, points, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        """ get_sine_position_encoding_points for single points.

        Args:
            points (tuple[int]): The temporal length, height and width of the window.
            num_pos_feats
            temperature
            normalize
            scale
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            mut_attn (bool): If True, add mutual attention to the module. Default: True
        """

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi


        y_embed, x_embed = points[:,:,0].unsqueeze(0), points[:,:, 1].unsqueeze(0)
        
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (self.window_size[0] + eps) * scale
            x_embed = x_embed / (self.window_size[1] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device='cuda')
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3)

        return pos_embed.squeeze(0)



    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        """ Get sine position encoding """
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32) - 1
        x_embed = not_mask.cumsum(2, dtype=torch.float32) - 1
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()

        