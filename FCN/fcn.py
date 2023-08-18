import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


def pad(x, h, w):
        diffY = h - x.size()[2]
        diffX = w - x.size()[3]
        return F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

class MultipleConv(nn.Module):
    def __init__(self, n_convs, in_channels, out_channels, kernel_size=3, padding=1, pooling = True):
        super(MultipleConv, self).__init__()
        
        self.pooling = pooling
        self.n_convs = n_convs
        self.maxpool = nn.MaxPool2d(2, 2)
        
        convs = []
        for i in range(n_convs):
            if i == 0: 
                convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(True)
                ))
            else:
                convs.append(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(True)
                ))
        self.convs = nn.ModuleList(convs)
    
    def forward(self, x):
        for i in range(self.n_convs):
            x = self.convs[i](x)
            
        if self.pooling:
            x = self.maxpool(x)
        
        return x
    
class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Upscale, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x1, x2):
        _, _, h, w = x2.size()
        x1 = F.relu6(self.convt(x1))
        x1 = pad(x1, h, w)
        return x1 + x2

        
class FCN(nn.Module):
    def __init__(self, n_class):
        super(FCN, self).__init__()
        
        self.n_class = n_class
        
        self.blocks = nn.ModuleList((
            MultipleConv(2, 3, 64),
            MultipleConv(2, 64, 128),
            MultipleConv(3, 128, 256),
            MultipleConv(3, 256, 512),
            MultipleConv(3, 512, 1024),
            MultipleConv(3, 1024, 1024, kernel_size=1, padding=0, pooling=False)
            
        ))
        
        self.conv1 = nn.Conv2d(256, 256, 1)
        self.conv2 = nn.Conv2d(512, 512, 1)
        
        self.upscale1 = Upscale(1024, 512, 2, 2)
        self.upscale2 = Upscale(512, 256, 2, 2)
        self.upscale3 = nn.Upsample(scale_factor=8, align_corners=True, mode='bilinear')
        
        self.outconv = nn.Conv2d(256, n_class, 1, padding=0)
        
    def forward(self, x):
        _, _, h, w = x.size()
        
        for i in range(3):
            x = self.blocks[i](x)
        feat1 = F.relu6(self.conv1(x))
        # 1/8 (256)
        
        x = self.blocks[3](x)
        feat2 = F.relu6(self.conv2(x))
        # 1/16 (512)
        
        x = self.blocks[4](x)
        x = self.blocks[5](x)
        # 1/32 (1024)

        x = F.relu6(self.upscale1(x, feat2))
        # 1/16 (512)
        
        x = F.relu6(self.upscale2(x, feat1))
        # 1/8 (256)
        
        x = F.relu6(self.upscale3(x))
        x = pad(x, h, w)
        
        x = self.outconv(x)
        return x