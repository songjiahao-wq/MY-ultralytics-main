# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 16:32
# @Author  : sjh
# @Site    : 
# @File    : import_module.py
# @Comment : 引用模块
#add ATT
from add.cv_attention import GAM_Attention
from add.cv_attention.EffectiveSE import EffectiveSEModule
from ultralytics.nn.models.add_models.blockii import BiLevelRoutingAttention, AttentionLePE, Attention
from ultralytics.nn.models.fighting_model.backbone.HorNet import HorNet
from ultralytics.nn.models.fighting_model.backbone.convnextv2 import convnextv2_att
from ultralytics.nn.models.add_models.my_attention import *
from ultralytics.nn.models.fighting_model.backbone.repghost import RepGhostBottleneck


from ultralytics.nn.models.fighting_model.attention.D_LKA_Attention import deformable_LKA_Attention_experimental, deformable_LKA_Attention

from ultralytics.nn.models.fighting_model.conv.ODConv import ODConv2d
from ultralytics.nn.models.fighting_model.conv.MBConv import MBConvBlock
from ultralytics.nn.models.fighting_model.conv.CondConv import CondConv
from ultralytics.nn.models.fighting_model.conv.DynamicConv import DynamicConv
from ultralytics.nn.models.fighting_model.conv.HorNet import gnconv  # use: gnconv(dim), c1 = c2
