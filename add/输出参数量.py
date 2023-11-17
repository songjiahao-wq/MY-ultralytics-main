import torch
import torch.nn as nn
from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv
from models.Models.Attention.my_attention import *
from models.Models.Attention.MultiScaleAttention import *
class GAMAttention(nn.Module):
    # GAM 注意力https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        return out
m1 = MixConv2d(128, 256, (3, 5), 1)
m2 = GAMAttention(256, 256)
results = profile(input=torch.randn(1, 256, 40, 40), ops=[m1, m2], n=3)




# Example usage
# input_tensor = torch.randn(8, 256, 32, 32)  # batch_size=8, in_channels=256, height=32, width=32
# model = FeatureEnhancementModule(in_channels=256)
# output_tensor = model(input_tensor)

