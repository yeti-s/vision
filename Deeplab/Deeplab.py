import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        
        return self.maxpool(x)

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, act=True) -> None:
        super(ConvBnRelu, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act
    
    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act:
            x = F.relu(x)
            
        return x
        

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, dilation=1) -> None:
        super(Bottleneck, self).__init__()
        
        self.conv1 = ConvBnRelu(in_channels, mid_channels, 1, 1, 0, 1, act=True)
        self.conv2 = ConvBnRelu(mid_channels, mid_channels, 3, stride, dilation, dilation, act=True)
        self.conv3 = ConvBnRelu(mid_channels, out_channels, 1, 1, 0, 1, act=False)
        self.need_short_conv = in_channels != out_channels or stride > 1
        if self.need_short_conv:
            self.short_conv = ConvBnRelu(in_channels, out_channels, 1, stride, 0, 1, act=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.need_short_conv:
            x = self.short_conv(x)
        
        return F.relu(out + x)
    
class ResBlock(nn.Module):
    def __init__(self, n_layers, in_channels, mid_channels, out_channels, stride, dilation) -> None:
        super(ResBlock, self).__init__()
        
        blocks = []
        for i in range(n_layers):
            layer_in_channels = in_channels if i == 0 else out_channels # first block only
            layer_stride = 1 if i < n_layers - 1 else stride            # last block only
            blocks.append(
                Bottleneck(layer_in_channels, mid_channels, out_channels, layer_stride, dilation)
            )
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x


class ResBody(nn.Module):
    def __init__(self, stem_channels, n_layers=[3,4,23,3]) -> None:
        super(ResBody, self).__init__()
        
        last_out_channels = stem_channels * 4
        blocks = [ResBlock(n_layers[0], stem_channels, stem_channels, last_out_channels, 2, 1)]
        for n in range(1, len(n_layers)):
            block_dilation = 1 if n < len(n_layers) - 1 else 2     # dilation last block only
            block_stride = 2 if n < len(n_layers) - 2 else 1        # no downsampling on last 2 blocks
            blocks.append(ResBlock(n_layers[n], last_out_channels, last_out_channels//2, last_out_channels*2, block_stride, block_dilation))
            last_out_channels *= 2
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        return x
    
class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, batch_norm=True, act=True) -> None:
        super(SeparableConv, self).__init__()
        
        mid_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride=stride, padding=dilation, dilation=dilation, groups=in_channels),
            nn.Conv2d(mid_channels, out_channels, 1)
        )
        self.batch_norm = batch_norm
        self.act = act
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        
    
    def forward(self, x):
        x = self.conv(x)
        
        if self.batch_norm:
            x = self.bn(x)
        
        if self.act:
            x = F.relu(x)
        
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1) -> None:
        super(XceptionBlock, self).__init__()
        
        self.conv = nn.Sequential(
            SeparableConv(in_channels, out_channels, dilation=dilation),
            SeparableConv(out_channels, out_channels, dilation=dilation),
            SeparableConv(out_channels, out_channels, stride=stride, dilation=dilation, act=False)
        )
        self.need_short_conv = in_channels != out_channels
        if self.need_short_conv:
            self.short_conv = ConvBnRelu(in_channels, out_channels, 1, stride=2, act=False)
    
    def forward(self, x):
        out = self.conv(x)
        if self.need_short_conv:
            x = self.short_conv(x)
        
        return F.relu(out + x)
    
class XceptionBody(nn.Module):
    def __init__(self, stem_channels) -> None:
        super().__init__()
        
        unit_channels = stem_channels * 4 # 256
        
        blocks = [
            XceptionBlock(stem_channels, stem_channels*2, 2),
            XceptionBlock(stem_channels*2, unit_channels, 2),
            XceptionBlock(unit_channels, unit_channels*3, 2)
        ]
        for _ in range(16):
            blocks.append(XceptionBlock(unit_channels*3, unit_channels*3))
        blocks.append(XceptionBlock(unit_channels*3, unit_channels*4, 2))
        blocks.append(SeparableConv(unit_channels*4, unit_channels*6))
        blocks.append(SeparableConv(unit_channels*6, unit_channels*6))
        blocks.append(SeparableConv(unit_channels*6, unit_channels*8))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x
        
    

class ImagePool(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.imagePool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1)
        )
        
    def forward(self, x):
        _, _, h, w = x.size()
        x = self.imagePool(x)
        
        return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
         

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ASPP, self).__init__()
        
        mid_channels = in_channels // 8
        
        self.layers = nn.ModuleList([
            ConvBnRelu(in_channels, mid_channels, 1, 1, 0, 1),
            ConvBnRelu(in_channels, mid_channels, 3, 1, padding=6, dilation=6),
            ConvBnRelu(in_channels, mid_channels, 3, 1, padding=12, dilation=12),
            ConvBnRelu(in_channels, mid_channels, 3, 1, padding=18, dilation=18),
            ImagePool(in_channels, mid_channels)
        ])
        self.conv = ConvBnRelu(mid_channels*5, out_channels, 1)
        
    def forward(self, x):
        x = torch.cat([layer(x) for layer in self.layers], dim=1)
        
        return self.conv(x)
        
        
        

class DeeplabV3(nn.Module):
    def __init__(self, in_channels, stem_channels, is_res):
        super(DeeplabV3, self).__init__()
        
        last_out_channels = stem_channels * 32
            
        self.stem = Stem(in_channels, stem_channels)
        self.body = ResBody(stem_channels) if is_res else XceptionBody(stem_channels)
        self.aspp = ASPP(last_out_channels, last_out_channels // 8)
    
    def forward(self, x):
        stem_out = self.stem(x)
        x = stem_out
        x = self.body(x)
        x = self.aspp(x)

        return x, stem_out
    
class DeeplabV3PlusSeg(nn.Module):
    def __init__(self, in_channels, n_classes, is_res):
        super(DeeplabV3PlusSeg, self).__init__()
        
        low_lev_feat_channels = 48
        stem_channels = 64
        self.deeplab = DeeplabV3(in_channels, stem_channels, is_res)
        # decoder
        self.conv1 = ConvBnRelu(stem_channels, low_lev_feat_channels, 1)
        self.conv2 = nn.Sequential(
            ConvBnRelu(stem_channels*4 + low_lev_feat_channels, stem_channels*4, 3, padding=1),
            ConvBnRelu(stem_channels*4, stem_channels*4, 3, padding=1),
            nn.Conv2d(stem_channels*4, n_classes, 1)
        )
    
    def forward(self, x):
        _, _, h, w = x.size()
        
        aspp_out , stem_out = self.deeplab(x)
        
        _, _, s_h, s_w = stem_out.size()
        low_lev_feats = self.conv1(stem_out)
        
        feats = F.interpolate(aspp_out, (s_h, s_w), mode='bilinear', align_corners=True)
        feats = self.conv2(torch.cat([low_lev_feats, feats], dim=1))
        
        return F.interpolate(feats, (h, w), mode='bilinear', align_corners=True)
    
    
# x = torch.rand(2, 3, 480, 360).cuda()
# # model = DeeplabV3(3).cuda()
# # model.eval()
# # with torch.no_grad():
# #     print(model(x).shape)
        
# model = DeeplabV3Seg(3, 10, True).cuda()
# model.eval()
# with torch.no_grad():
#     print(model(x).shape)