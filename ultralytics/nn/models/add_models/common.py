#RFA exp start********************************
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torchvision.ops import deform_conv2d
from mmcv.ops import DeformConv2d
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class DeformCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(DeformCAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        # Deformable Convolution
        self.offset_conv = nn.Conv2d(inp, 2 * kernel_size * kernel_size, kernel_size=1, stride=2, padding=0)
        self.deform_conv = DeformConv2d(inp, oup, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)
        # self.deform_conv = torchvision.ops.deform_conv2d(inp, oup, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)
        self.bn2 = nn.BatchNorm2d(oup)
        self.silu = nn.SiLU()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        # Deformable Convolution
        offset = self.offset_conv(out)
        out = self.deform_conv(out, offset)
        out = self.bn2(out)
        out = self.silu(out)

        return out
class DeformCBAMConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // reduction, inp, 1, bias=False)
        )

        self.spatital = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                  padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # Deformable Convolution
        self.offset_conv = nn.Conv2d(inp, 2 * kernel_size * kernel_size, kernel_size=1, stride=2, padding=0)
        self.deform_conv = DeformConv2d(inp, oup, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)
        # self.deform_conv = torchvision.ops.deform_conv2d(inp, oup, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)
        self.bn2 = nn.BatchNorm2d(oup)
        self.silu = nn.SiLU()

    def forward(self, x):
        identity = x
        out = x
        # Deformable Convolution
        offset = self.offset_conv(out)
        out = self.deform_conv(out, offset)
        out = self.bn2(out)
        out = self.silu(out)

        return out