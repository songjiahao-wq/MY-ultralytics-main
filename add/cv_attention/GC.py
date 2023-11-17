# import torch
# from torch import nn as nn
# import torch.nn.functional as F
# from timm.add_models.layers.create_act import create_act_layer, get_act_layer
# from timm.add_models.layers.helpers import make_divisible
# from timm.add_models.layers.mlp import ConvMlp
# from timm.add_models.layers.norm import LayerNorm2d
#
#
# class GlobalContext(nn.Module):
#
#     def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
#                  rd_ratio=1./8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
#         super(GlobalContext, self).__init__()
#         act_layer = get_act_layer(act_layer)
#
#         self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
#
#         if rd_channels is None:
#             rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
#         if fuse_add:
#             self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
#         else:
#             self.mlp_add = None
#         if fuse_scale:
#             self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
#         else:
#             self.mlp_scale = None
#
#         self.gate = create_act_layer(gate_layer)
#         self.init_last_zero = init_last_zero
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.conv_attn is not None:
#             nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
#         if self.mlp_add is not None:
#             nn.init.zeros_(self.mlp_add.fc2.weight)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         if self.conv_attn is not None:
#             attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
#             attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
#             context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
#             context = context.view(B, C, 1, 1)
#         else:
#             context = x.mean(dim=(2, 3), keepdim=True)
#
#         if self.mlp_scale is not None:
#             mlp_x = self.mlp_scale(context)
#             x = x * self.gate(mlp_x)
#         if self.mlp_add is not None:
#             mlp_x = self.mlp_add(context)
#             x = x + mlp_x
#
#         return x
#
# if __name__ == '__main__':
#     input=torch.randn(50,512,7,7)
#     gc = GlobalContext(512)
#     output=gc(input)
#     print(output.shape)
import torch
import torch.nn as nn
import torchvision


class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


if __name__=='__main__':
    model = GlobalContextBlock(inplanes=16, ratio=0.25)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)