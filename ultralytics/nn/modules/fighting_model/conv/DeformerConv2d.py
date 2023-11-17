import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x


if __name__ == "__main__":
    deformable_conv2d = DeformableConv2d(in_dim=3, out_dim=4, kernel_size=1, offset_groups=3, with_mask=False)
    print(deformable_conv2d(torch.randn(1, 3, 5, 7)).shape)

    deformable_conv2d = DeformableConv2d(in_dim=3, out_dim=6, kernel_size=1, groups=3, offset_groups=3, with_mask=True)
    print(deformable_conv2d(torch.randn(1, 3, 5, 7)).shape)

"""
torch.Size([1, 4, 5, 7])
torch.Size([1, 6, 5, 7])
"""