import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from models.experimental import MixConv2d
from torch.nn import init
"""
#num_scales参数决定了注意力机制将考虑的不同尺度的数量。这个值取决于你想在你的模型中捕获的多尺度背景的水平和输入图像的大小。
#没有一个放之四海而皆准的数值，因为它取决于你正在处理的具体问题和数据集。
#对于CIFAR-10数据集，它有32x32的图像，你可以从num_scales的值开始，等于3。
#这将允许注意力机制使用扩张的卷积在三种不同的感受野尺寸（3x3，5x5和7x7）上捕捉上下文。
"""
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = in_channels // reduction_ratio

        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.mid_channels, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)
        attention = self.conv1(x)
        attention = self.bn1(attention)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention_map = self.sigmoid(attention)

        # Apply attention map to input
        out = x * attention_map
        return out
class LightweightMultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7], reduction_ratio=16):
        super(LightweightMultiScaleSpatialAttention, self).__init__()

        self.branches = nn.ModuleList()
        mid_channels = in_channels // reduction_ratio

        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=mid_channels),
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)

        # Apply different kernel sizes
        attention_maps = []
        for branch in self.branches:
            attention_map = branch(x)
            attention_map = torch.sigmoid(attention_map)
            attention_maps.append(attention_map)

        # Element-wise sum of attention maps
        combined_attention_map = torch.sum(torch.stack(attention_maps), dim=0)

        # Apply the combined attention map to the input
        out = x * combined_attention_map

        return out
class SpatialMultiScaleAttention(nn.Module):
    def __init__(self, in_channels, num_scales, groups= [4, 8, 16]):
        super(SpatialMultiScaleAttention, self).__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        mid_channels = in_channels // 16
        self.sigmoid = nn.Sigmoid()
        # Define the attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_scales): #MixConv2d(128, 256, (3, 5), 1)
            # self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1))
            self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1, groups=groups[i]))
            # self.attention_layers.append(MixConv2d(in_channels, in_channels, k=(3, 5), s=1))# 1

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()
        temp = x
        # Calculate attention maps for each scale
        attention_maps = []
        out = []
        for i in range(self.num_scales):
            attention_map = self.attention_layers[i](temp)
            temp = attention_map
            # attention_map = F.softmax(attention_map.view(batch_size, -1), dim=1)
            attention_map = self.sigmoid(attention_map.view(batch_size, -1))
            attention_map = attention_map.view(batch_size, self.in_channels, height, width)
            attention_maps.append(attention_map)

        # Sum attention maps
        attention_maps = sum(attention_maps)

        # Apply attention to the input
        x = x * attention_maps
        return x
class Spatial_Test2(nn.Module):
    def __init__(self, in_channels, num_scales,groups= [4, 8, 16]):
        super(Spatial_Test2, self).__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        mid_channels = in_channels // 16
        self.sigmoid = nn.Sigmoid()
        # Define the attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_scales): #MixConv2d(128, 256, (3, 5), 1)
            # self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1))
            # self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1, groups=groups[i]))
            # self.attention_layers.append(MixConv2d(in_channels, in_channels, k=(3, 5), s=1))# 1
            if i == 1:
                self.attention_layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1))  # 2
            else:
                self.attention_layers.append(nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1))  # 2

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()

        # Calculate attention maps for each scale
        attention_maps = []
        out_maps = []
        for i in range(self.num_scales):
            attention_map = self.attention_layers[i](x)
            attention_map = F.softmax(attention_map.view(batch_size, -1), dim=1)
            attention_map = attention_map.view(batch_size, self.in_channels, height, width)
            attention_maps.append(attention_map)

        for i in range(self.num_scales):
            out_maps.append(x * attention_maps[i])
        # Sum attention maps
        attention_maps = torch.cat(out_maps,dim=1)

        # Apply attention to the input
        x = x * attention_maps
        return x
class Spatial_Test1(nn.Module):
    def __init__(self, in_channels, num_scales,groups= [4, 8, 16]):
        super(Spatial_Test1, self).__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        mid_channels = in_channels // 16
        self.sigmoid = nn.Sigmoid()
        # Define the attention layers
        self.attention_layers = nn.ModuleList()
        self.cv1 = Conv(num_scales * mid_channels, in_channels, k=1)
        for i in range(num_scales): #MixConv2d(128, 256, (3, 5), 1)
            # self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1))
            # self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1, groups=groups[i]))
            # self.attention_layers.append(MixConv2d(in_channels, in_channels, k=(3, 5), s=1))# 1
            if i == 1:
                self.attention_layers.append(MixConv2d(in_channels, mid_channels, k=(3, 5), s=1))  # 2
            else:
                self.attention_layers.append(MixConv2d(mid_channels, mid_channels, k=(3, 5), s=1))  # 2

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()

        # Calculate attention maps for each scale
        attention_maps = []
        for i in range(self.num_scales):
            attention_map = self.attention_layers[i](x)
            attention_map = F.softmax(attention_map.view(batch_size, -1), dim=1)
            attention_map = attention_map.view(batch_size, self.in_channels, height, width)
            attention_maps.append(attention_map)

        # Sum attention maps
        attention_maps = torch.cat(attention_maps,dim=1)
        attention_maps = self.cv1(attention_maps)
        # Apply attention to the input
        x = x * attention_maps
        return x
class ChannelAttentionModified(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModified, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / reduction_ratio), in_channels),
        )
    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)

        channel_att_avg = self.fc(avg_pool).view(batch_size, channels, 1, 1)
        channel_att_max = self.fc(max_pool).view(batch_size, channels, 1, 1)

        channel_att = self.sig(channel_att_avg + channel_att_max)
        return x * channel_att
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        #squeeze() 函数用于删除张量中尺寸为 1 的维度
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        y = y.expand_as(x)
        return x * y
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, out_channels,  num_scales):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.spatial_attention = SpatialMultiScaleAttention(in_channels, num_scales)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        # self.spatial_attention = LightweightMultiScaleSpatialAttention(in_channels)
        # self.channel_attention = ChannelAttentionModified(in_channels)
        self.channel_attention = ECAAttention()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # x = F.relu(self.conv1(x))

        weight = self.spatial_attention(x) + self.channel_attention(x)
        # weight =  self.channel_attention(x)
        # weight = self.spatial_attention(weight)
        out_feature = weight + x #1 0.809
        # out_feature = weight * x #2
        # out_feature = weight * (1 + x) #3 加残差
        # out_feature = weight  #4 加残差
        return out_feature

class MultiScaleAttentionCSPA(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales):
        super(MultiScaleAttentionCSPA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.spatial_attention = SpatialMultiScaleAttention(in_channels, num_scales)
        self.channel_attention = ChannelAttentionModified(in_channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.cv1 = Conv(in_channels,in_channels,1,1)
        self.cv2 = Conv(in_channels,in_channels,1,1)
    def forward(self, x):
        y1 =self.cv1(x)
        y2 = self.cv2(x)
        weight = self.spatial_attention(y1) + self.channel_attention(y1)
        out_feature = weight + y2  # 1 0.809
        return out_feature

class MultiScaleAttentionCSPB(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales):
        super(MultiScaleAttentionCSPB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.spatial_attention = SpatialMultiScaleAttention(in_channels, num_scales)
        self.channel_attention = ChannelAttentionModified(in_channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.cv1 = Conv(in_channels, in_channels, 1, 1)
        self.cv2 = Conv(in_channels, in_channels, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        weight = self.spatial_attention(x) + self.channel_attention(x)
        out_feature = weight + y1  #
        return out_feature

class MultiScaleAttentionCSPC(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales):
        super(MultiScaleAttentionCSPC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.spatial_attention = SpatialMultiScaleAttention(in_channels, num_scales)
        self.channel_attention = ChannelAttentionModified(in_channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.cv1 = Conv(in_channels, in_channels, 1, 1)
        self.cv2 = Conv(in_channels, in_channels, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        weight = self.spatial_attention(y1) + self.channel_attention(y1)
        out_feature = weight + y1  #
        return out_feature
#解释
    """
    #在这个特定的例子中，空间注意力机制 (self.spatial_attention(x)) 和通道注意力机制 (self.channel_attention(x))
    #都输出一个与输入特征图 (x) 形状相同的张量。这两个输出张量中的每个元素都是一个注意力权重，它们分别表示空间和通道维度上的注意力分布。
    #逐元相乘操作将这两个输出张量相乘，从而将空间和通道注意力权重结合在一起。这样，我们可以在一个操作中同时捕捉空间和通道维度上的上下文信息。
    #最后，将注意力权重逐元素地应用于输入特征图 x。这意味着每个特征图的值都会根据空间和通道注意力权重进行调整。
    #这样，经过注意力机制调整的特征图能更好地捕捉输入数据中的重要信息。
    """
if __name__ == "__main__":
    input = torch.randn(1, 128, 20, 20)
    models = MultiScaleAttention(128, 128, 1)
    print(models(input).shape)
"""
1. 空间多尺度注意力机制 (Spatial Multi-Scale Attention):

空间注意力机制关注的是输入特征图中每个位置的重要性。在这个例子中，我们实现了一个空间多尺度注意力机制，它可以捕获不同尺度下的上下文信息。这种注意力机制通过在不同空间范围内进行局部邻域操作来实现多尺度特性。在这个实现中，我们使用了具有不同 dilation 参数的卷积层。这样可以在不增加计算复杂度的情况下，扩大感受野。

对于每个尺度，我们计算一个注意力图。这些注意力图随后相加，从而形成一个综合的空间注意力图。通过逐元素乘法，将空间注意力图应用于输入特征图，从而突出显示在不同尺度下具有较高权重的区域。

2. 通道注意力机制 (Channel Attention):

通道注意力机制关注的是输入特征图中每个通道的重要性。在上述实现中，我们提出了一种修改后的通道注意力机制，它基于全局平均池化和全局最大池化，以及一个全连接层。与 Squeeze-and-Excitation (SE) 块相比，它减少了参数数量，从而实现了轻量化。

这个通道注意力机制首先计算输入特征图的全局平均池化和全局最大池化，然后将它们分别输入一个全连接层。这个全连接层将通道数量降低为原始通道数量除以一个固定的缩减比例（例如 16）。将平均池化和最大池化得到的两个注意力向量相加，从而形成一个综合的通道注意力向量。将通道注意力向量应用于输入特征图的通道维度，突出显示具有较高权重的通道。

3. 注意力机制的结合：

在上述实现的神经网络模型中，我们将空间多尺度注意力机制和修改后的通道注意力机制结合在一起。这是通过将两个注意力机制的输出逐元素相乘来实现的。这样，我们可以同时捕捉空间和通道维度上的上下文信息，从而使模型更好地关注输入数据中的重要特征。这种组合在某些情况下可能比单独使用空间或通道注意力机制产生更好的性能。
"""
"""
以下是上述注意力机制的理论公式：
空间多尺度注意力机制 (Spatial Multi-Scale Attention):
对于每个尺度 i (i = 1, 2, ..., num_scales)，我们计算空间注意力图 A_s_i 如下：
A_s_i = f_s_i(x)
其中 f_s_i(x) 是具有不同 dilation 参数的卷积运算，用于扩展感受野以捕获第 i 个尺度的上下文信息。
接下来，将所有尺度的空间注意力图相加，得到综合空间注意力图 A_s：
A_s = Σ A_s_i
通道注意力机制 (Channel Attention):
计算全局平均池化 (GAP) 和全局最大池化 (GMP)：
GAP_x = GAP(x)
GMP_x = GMP(x)
将 GAP_x 和 GMP_x 分别输入一个全连接层 f_c(x)，得到通道注意力向量 A_c_avg 和 A_c_max：
A_c_avg = f_c(GAP_x)
A_c_max = f_c(GMP_x)
将 A_c_avg 和 A_c_max 相加，得到综合通道注意力向量 A_c：
A_c = A_c_avg + A_c_max
注意力机制的结合：
将空间多尺度注意力图 A_s 和通道注意力向量 A_c 结合起来，形成一个综合注意力图 A：
A = A_s ⊙ A_c
其中 ⊙ 表示逐元素乘法。这个综合注意力图 A 既包含空间维度上的信息，也包含通道维度上的信息。
将注意力图 A 应用于输入特征图 x，得到输出特征图 y：
y = x ⊙ A
在这个实现中，我们直接输出经过注意力调整的特征图 y。
"""

class MultiScaleFrequencyAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_sizes=[1, 2, 4]):
        super().__init__()

        # 定义卷积层和池化层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in pool_sizes])
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels // reduction_ratio, in_channels),
                nn.Sigmoid()
            ) for _ in pool_sizes
        ])

    def forward(self, x):
        # 计算频域特征向量
        x_fft = torch.fft.rfft(x)
        x_abs = (x_fft ** 2).sum(dim=-1).sqrt()

        # 计算每个尺度下的注意力权重
        attention_weights = []
        for pool, fc in zip(self.pools, self.fcs):
            y = pool(x_abs)
            y = torch.flatten(y, start_dim=1)
            y = fc(y)
            attention_weights.append(y)

        # 将不同尺度下的注意力权重相加
        attention_weight = sum(attention_weights)
        attention_weight = attention_weight.unsqueeze(-1).unsqueeze(-1)

        # 将注意力权重应用到原始信号上
        x = x_abs * attention_weight
        y = torch.irfft(x, signal_ndim=2, signal_sizes=x_fft.size()[2:])

        return y
class FrequencyAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        # 定义卷积层和池化层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算频域特征向量
        x_fft = torch.rfft(x, signal_ndim=2)
        x_abs = (x_fft ** 2).sum(dim=-1).sqrt()
        y = self.conv(x_abs)
        y = self.pool(y)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        # 计算注意力权重并应用到原始信号上
        attention_weight = y.unsqueeze(-1).unsqueeze(-1)
        x = x * attention_weight

        return x


class MultiScaleFreqAttention(nn.Module):
    def __init__(self, num_scales, pool_sizes=[3, 5, 7]):
        super(MultiScaleFreqAttention, self).__init__()
        self.num_scales = num_scales
        self.pool_sizes = pool_sizes

        self.pool_layers = nn.ModuleList()
        for pool_size in pool_sizes:
            self.pool_layers.append(nn.AdaptiveAvgPool2d((1, pool_size)))

    def forward(self, x):
        # x: input tensor of shape (batch_size, channels, height, width)
        x_freq = torch.fft.rfft2(x)  # Transform the input tensor to the frequency domain

        attention_maps = []
        for pool in self.pool_layers:
            attention_map = pool(torch.abs(x_freq))
            attention_map = attention_map / torch.sum(attention_map, dim=(-1, -2), keepdim=True)
            attention_maps.append(attention_map)

        attention_maps = sum(attention_maps)
        x_freq_weighted = x_freq * attention_maps.unsqueeze(-1)

        x_weighted = torch.fft.irfft2(x_freq_weighted)  # Transform the weighted tensor back to the time domain
        return x_weighted

class SoftThresholdAttentionResidual(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, threshold=0.5):
        super(SoftThresholdAttentionResidual, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.GLU = nn.GLU()
        self.sigmoid = nn.Sigmoid()
        self.threshold = nn.Parameter(torch.tensor([threshold]))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        threshold = self.threshold.view(1, -1, 1, 1)
        y_thresh = torch.where(y < threshold, torch.zeros_like(y), torch.ones_like(y))
        return x * y_thresh.expand_as(x) #1
        # return  y_thresh * x #2
        # return  self.bias + y_thresh * x #3
        # return  self.GLU(self.bias + y_thresh * x) #4
        return  x + self.GLU(self.bias + y_thresh * x) #5