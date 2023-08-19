from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.ops as ops

from torch.nn.common_types import _size_2_t
from torch.nn.init import xavier_uniform_, constant_

class TorchDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False):
        super(TorchDeformConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * in_channels * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulation_conv = nn.Conv2d(in_channels,
                        in_channels * kernel_size * kernel_size,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=self.padding,
                        bias=True)
        # nn.init.constant_(self.modulation_conv.weight, 1.0)
        # nn.init.constant_(self.modulation_conv.bias, 0.0)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


    def forward(self, x):
        _, _, h, w = x.size()
        max_offset = max(h, w) / 4 

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulation = self.modulation_conv(x)
        x = ops.deform_conv2d(x, offset, self.conv.weight, self.conv.bias, padding=self.padding, mask=modulation)

        return x

class DeformConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: _size_2_t = 1, padding: _size_2_t = 0, bias=True) -> None:
        super(DeformConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, padding=padding, bias=bias)

        self.kernel_size = kernel_size
        N = kernel_size ** 2

        self.offset_conv = nn.Conv2d(in_channels, 2*N, kernel_size, stride=stride, padding=1, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.modulation_conv = nn.Conv2d(in_channels, N, kernel_size, stride=stride, padding=1, bias=True)
        # nn.init.constant_(self.offset_conv.weight, 0.5)
        # nn.init.constant_(self.offset_conv.bias, 0)


    def forward(self, x):
        b, c, h, w = x.size()
        ks = self.kernel_size
        N = ks ** 2

        # (b, 2N, h, w)
        offset = self.offset_conv(x)

        # (b, N, h, w, 2)
        coords = offset + self.create_grid(h, w).to(offset.device)
        coords = coords.view(b, N, h, w, 2)
        coords[...,0].clamp_(min=0, max=h-1)           # offset_y
        coords[...,1].clamp_(min=0, max=w-1)           # offset_x

        # (b, N, h, w)
        d_l = coords[...,1] - coords[...,1].floor()
        d_t = coords[...,0] - coords[...,0].floor()
        d_r = 1 - d_l
        d_b = 1 - d_t

        modulation = nn.functional.sigmoid(self.modulation_conv(x))
        
        # (b, 1, N, h, w)
        d_l = d_l.unsqueeze(1)
        d_t = d_t.unsqueeze(1)
        d_r = d_r.unsqueeze(1)
        d_b = d_b.unsqueeze(1)

        modulation = modulation.unsqueeze(1)

        # (b, N, h, w, 2)
        lt = torch.stack((coords[...,0].floor(), coords[...,1].floor()), -1)
        rb = torch.stack((coords[...,0].ceil(), coords[...,1].ceil()), -1)
        lb = torch.stack((rb[...,0], lt[...,1]), -1)
        rt = torch.stack((lt[...,0], rb[...,1]), -1)
        # (b, c, N, h, w)
        lt = self.get_values(x, lt.long())
        rb = self.get_values(x, rb.long())
        lb = self.get_values(x, lb.long())
        rt = self.get_values(x, rt.long())

        grid = d_t*(d_r*rb+d_l*lb) + d_b*(d_r*lt+d_l*rt)
        grid = grid * modulation

        # (b, c, h * ks, w * ks)
        grid = grid.contiguous().view(b, c, h * ks, w * ks)

        return nn.functional.relu(super(DeformConv2d, self).forward(grid))

    def create_grid(self, height, width):
        grid = np.meshgrid(range(width), range(height))
        return torch.from_numpy(grid[0]).type(torch.float32)
    
    # x(b, c, h, w),  grid(b, N, h, w, 2), return(b, c, N, h, w)
    def get_values(self, x, grid):
        c = x.size(1)
        b, N, h, w, _ = grid.size()

        # (b, c, N, h * w)
        x = x.view(b, c, -1)
        x = x.unsqueeze(2).tile((1,1,N,1))

        grid = grid[...,0] * w + grid[...,1]
        grid = grid.view(b, N, -1)
        grid = grid.unsqueeze(1).tile((1,c,1,1))

        # (b, c, N, h, w)
        values = x.gather(-1, grid)
        values = values.view(b, c, N, h, w)
        return values


class DeformConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias=True) -> None:
        super(DeformConv, self).__init__()

        self.kernel_size = kernel_size
        N = kernel_size ** 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, padding=padding, bias=bias)

        self.offset_conv = nn.Conv2d(in_channels, 2*N, kernel_size, stride=stride, padding=1, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.modulation_conv = nn.Conv2d(in_channels, N, kernel_size, stride=stride, padding=1, bias=True)
        # nn.init.constant_(self.offset_conv.weight, 1.0)
        # nn.init.constant_(self.offset_conv.bias, 0)


    def forward(self, x):
        b, c, h, w = x.size()
        ks = self.kernel_size
        N = ks ** 2

        # (b, 2N, h, w)
        offset = self.offset_conv(x)
        # (b, N, h, w)
        x_offset = offset[:,:N,...]
        y_offset = offset[:,N:,...]
        
        # create grid
        index = torch.meshgrid([torch.linspace(-1.0, 1.0, h, device=offset.device), torch.linspace(-1.0, 1.0, w, device=offset.device)])
        x_index = index[1]
        y_index = index[0]

        # (b, N, h, w)
        x_coord = (x_offset + x_index).clamp_(-1.0, 1.0)
        y_coord = (y_offset + y_index).clamp_(-1.0, 1.0)
        
        coord = torch.stack((x_coord, y_coord), dim=-1)
        # (N, b, c, h, w, 2)
        coord = coord.permute(1,0,2,3,4)

        interpolated = torch.stack([nn.functional.grid_sample(x, coord[i], mode='bilinear', align_corners=True) for i in range(9)], dim=1)
    
        # (c, b, N, h, w)
        interpolated = interpolated.permute(2, 0, 1, 3, 4)
        modulation = self.modulation_conv(x)
        interpolated = interpolated * modulation
        interpolated = interpolated.permute(1, 0, 2, 3, 4)
        
        interpolated = torch.cat([torch.cat([interpolated[:,:,(i+j*3),...] for i in range(3)], dim=-1) for j in range(3)], dim=2)
        return nn.functional.conv2d(interpolated, self.conv.weight, self.conv.bias, dilation=(h+1, w+1), padding=self.conv.padding)

class DefromConvV3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, kernel_size: int = 3, stride: _size_2_t = 1, padding: _size_2_t = 1, bias=True) -> None:
        super(DefromConvV3, self).__init__()
        
        self.in_channels = in_channels
        self.out_channles = out_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.N = kernel_size**2
        
        self.offset_conv = nn.Conv2d(in_channels, 2*self.N*groups, kernel_size, stride, padding, bias=True)
        self.mask_conv = nn.Conv2d(in_channels, self.N*groups, kernel_size, stride, padding, bias=True)
        self.in_proj = nn.Linear(in_channels, in_channels)
        # self.out_proj = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, groups=groups)
        
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=True)
        
    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        input = self.in_proj(x.permute(0, 2, 3, 1))
        input = input.permute(0, 3, 1, 2)
        
        # depth-wise
        x1 = self.dw_conv(x)
        x1 = F.layer_norm(x1, x1.shape[1:])
        x1 = F.gelu(x1)
        
        offset = self.offset_conv(x1)
        mask = self.mask_conv(x1).reshape(b, h, w, self.groups, -1)
        mask = F.softmax(mask, -1).reshape(b, -1, h, w)
    
        
        x = ops.deform_conv2d(input, offset, self.out_proj.weight, self.out_proj.bias, padding=self.padding, mask=mask)
        return x
        

# test = DeformConv2d(3, 6, 3, padding=1)
# a = torch.rand(32, 3, 16, 24)
# test(a)

test = DefromConvV3(32, 32, 4)
a = torch.rand(64, 32, 16, 24)
test(a)

#https://github.com/4uiiurz1/pytorch-deform-conv-v2/blob/master/deform_conv_v2.py
#https://github.com/oeway/pytorch-deform-conv/blob/master/torch_deform_conv/deform_conv.py