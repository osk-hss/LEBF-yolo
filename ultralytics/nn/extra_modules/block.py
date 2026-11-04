# Ultralytics YOLO, AGPL-3.0 license
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from ..modules.conv import Conv
from ..modules.block import C3k, C3k2


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
        
    
    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)


class Conv_Extra(nn.Module):
    def __init__(self, channel):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(Conv(channel, 64, 1),
                                   Conv(64, 64, 3),
                                   Conv(64, channel, 1, act=False))
    def forward(self, x):
        out = self.block(x)
        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size=5, sigma=0.5, feature_extra=True):
        super().__init__()
        assert size % 2 == 1, "Gaussian kernel size must be odd."
        assert sigma > 0, "Gaussian sigma must be positive."
        self.feature_extra = feature_extra
        self.size = size
        self.sigma = sigma
        gaussian = self.gaussian_kernel(size, sigma)
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=size // 2, groups=dim, bias=False)
        with torch.no_grad():
            self.gaussian.weight.copy_(gaussian.repeat(dim, 1, 1, 1))
        self.gaussian.weight.requires_grad_(False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        if feature_extra:
            self.conv_extra = Conv_Extra(dim)

    def forward(self, x):
        gaussian = self.act(self.norm(self.gaussian(x)))
        return self.conv_extra(x + gaussian) if self.feature_extra else gaussian

    def gaussian_kernel(self, size, sigma):
        radius = size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)


class LFEA(nn.Module):
    def __init__(self, channel):
        super(LFEA, self).__init__()
        self.channel = channel
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = self.block = Conv(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, c, att):
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(c + att * wei)

        return x


class LFE_Module(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0.1, gaussian_size=5, gaussian_sigma=0.5):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False))
        self.LFEA = LFEA(dim)
        self.gaussian = Gaussian(dim, gaussian_size, gaussian_sigma)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        att = self.gaussian(x)
        x_att = self.LFEA(x, att)
        return x + self.norm(self.drop_path(self.mlp(x_att)))


class C3k_LFEM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3, gaussian_size=5, gaussian_sigma=0.5):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(LFE_Module(c_, gaussian_size=gaussian_size, gaussian_sigma=gaussian_sigma) for _ in range(n)))


class C3k2_LFEM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, gaussian_size=5, gaussian_sigma=0.5):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LFEM(self.c, self.c, n, shortcut, g, gaussian_size=gaussian_size,
                                        gaussian_sigma=gaussian_sigma) if c3k else LFE_Module(self.c,
                                                                                              gaussian_size=gaussian_size,
                                                                                              gaussian_sigma=gaussian_sigma)
                               for _ in range(n))
