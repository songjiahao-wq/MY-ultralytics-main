# -*- coding: utf-8 -*-
# @Time    : 2024/9/3 19:42
# @Author  : sjh
# @Site    : 
# @File    : 111.py
# @Comment :
import torch
import torch.nn as nn

class BaseModule(nn.Module):
    """Placeholder for the actual BaseModule class which should be inherited."""
    def __init__(self):
        super(BaseModule, self).__init__()

class MSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,channels1,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        out = self.conv0(x)
        for i in range(len(self._modules) - 2):
            conv = getattr(self, f'conv{i//2}_1' if i % 2 == 0 else f'conv{i//2}_2')
            out = conv(out)
        out = self.conv3(out)
        return out

# 测试 MSCAAttention 模块
if __name__ == "__main__":
    # 创建MSCAAttention实例
    channels = 16  # 例如使用16个通道
    attention_module = MSCAAttention(channels,channels)

    # 创建输入张量 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, channels, 32, 32)  # 例如 1个样本，16个通道，32x32大小

    # 前向传播
    output_tensor = attention_module(input_tensor)

    # 打印输入和输出形状
    print("Input shape: ", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
