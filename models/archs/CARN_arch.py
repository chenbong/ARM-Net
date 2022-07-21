import torch
import torch.nn as nn
import torch.nn.functional as F
import math



import models.archs.arch_util as arch_util
from models.archs.arch_util import USConv2d



class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=1):

        m = [USConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, us=[True, True])]
        m.append(nn.ReLU(True))
        super(BasicBlock, self).__init__(*m)

class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            USConv2d(in_channels, out_channels, 3, 1, 1, groups=group, us=[True, True]),
            nn.ReLU(inplace=True),
            USConv2d(out_channels, out_channels, 3, 1, 1, groups=group, us=[True, True]),
            nn.ReLU(inplace=True),
            USConv2d(out_channels, out_channels, 1, 1, 0, us=[True, True]),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class Block(nn.Module):
    def __init__(self, nf, group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(nf, nf, group=group)
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [USConv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group, us=[True, True]), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [USConv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group, us=[True, True]), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)

class US_CARN_M(nn.Module):
    def __init__(self, in_nc, out_nc, nf=64, scale=4, multi_scale=False, group=4):
        super(US_CARN_M, self).__init__()

        self.scale = scale
        rgb_range = 1
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.entry = USConv2d(in_nc, nf, 3, 1, 1, us=[False, True])

        self.b1 = Block(nf, group=group, )
        self.b2 = Block(nf, group=group, )
        self.b3 = Block(nf, group=group, )
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0, )
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0, )
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0, )
        
        self.upsample = UpsampleBlock(nf, scale=scale, multi_scale=multi_scale, group=group)
        self.exit = USConv2d(nf, out_nc, 3, 1, 1, us=[True, False])

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out