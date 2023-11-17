# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 16:32
# @Author  :
# @Site    : 
# @File    : import_module.py
# @Comment : 引用模块
# add ATT start------------------------------------
from ultralytics.nn.models.fighting_model.attention.A2Atttention import DoubleAttention
from ultralytics.nn.models.fighting_model.attention.ACmixAttention import ACmix
from ultralytics.nn.models.fighting_model.attention.AFT import AFT_FULL
from ultralytics.nn.models.fighting_model.attention.Axial_attention import AxialAttention, AxialImageTransformer
from ultralytics.nn.models.fighting_model.attention.BAM import BAMBlock
from ultralytics.nn.models.fighting_model.attention.Biformer import BiLevelRoutingAttention
from ultralytics.nn.models.fighting_model.attention.CBAM import CBAMBlock
from ultralytics.nn.models.fighting_model.attention.CoAtNet import CoAtNet
from ultralytics.nn.models.fighting_model.attention.CoordAttention import CoordAtt
from ultralytics.nn.models.fighting_model.attention.CrissCrossAttention import CrissCrossAttention
from ultralytics.nn.models.fighting_model.attention.D_LKA_Attention import deformable_LKA_Attention_experimental, \
    deformable_LKA_Attention
from ultralytics.nn.models.fighting_model.attention.DANet import DAModule
from ultralytics.nn.models.fighting_model.attention.DAT import DAT, LocalAttention
from ultralytics.nn.models.fighting_model.attention.ECAAttention import ECAAttention
from ultralytics.nn.models.fighting_model.attention.EMANet import EMAU
from ultralytics.nn.models.fighting_model.attention.EMSA import EMSA
from ultralytics.nn.models.fighting_model.attention.EVCBlock import EVCBlock
from ultralytics.nn.models.fighting_model.attention.ExternalAttention import ExternalAttention
from ultralytics.nn.models.fighting_model.attention.Focused_Linear_Attention import FocusedLinearAttention
from ultralytics.nn.models.fighting_model.attention.GAMAttention import GAMAttention
from ultralytics.nn.models.fighting_model.attention.gfnet import GFNet
from ultralytics.nn.models.fighting_model.attention.HaloAttention import HaloAttention
from ultralytics.nn.models.fighting_model.attention.LKA import LKA
from ultralytics.nn.models.fighting_model.attention.MOATransformer import MOATransformer
from ultralytics.nn.models.fighting_model.attention.MobileViTAttention import MobileViTAttention
from ultralytics.nn.models.fighting_model.attention.MobileViTv2Attention import MobileViTv2Attention
from ultralytics.nn.models.fighting_model.attention.MUSEAttention import MUSEAttention
from ultralytics.nn.models.fighting_model.attention.OutlookAttention import OutlookAttention
from ultralytics.nn.models.fighting_model.attention.ParNetAttention import ParNetAttention
from ultralytics.nn.models.fighting_model.attention.PSA import PSA
from ultralytics.nn.models.fighting_model.attention.ResidualAttention import ResidualAttention
from ultralytics.nn.models.fighting_model.attention.S2Attention import S2Attention
from ultralytics.nn.models.fighting_model.attention.SEAttention import SEAttention
from ultralytics.nn.models.fighting_model.attention.SGE import SpatialGroupEnhance
from ultralytics.nn.models.fighting_model.attention.ShuffleAttention import ShuffleAttention
from ultralytics.nn.models.fighting_model.attention.SimAM import SimAM
from ultralytics.nn.models.fighting_model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from ultralytics.nn.models.fighting_model.attention.SKAttention import SKAttention
from ultralytics.nn.models.fighting_model.attention.TripletAttention import TripletAttention
from ultralytics.nn.models.fighting_model.attention.UFOAttention import UFOAttention
from ultralytics.nn.models.fighting_model.attention.ViP import WeightedPermuteMLP
from ultralytics.nn.models.add_models.my_attention import *
# add ATT end ------------------------------------
from add.cv_attention.EffectiveSE import EffectiveSEModule
from ultralytics.nn.models.fighting_model.backbone.HorNet import HorNet
from ultralytics.nn.models.fighting_model.backbone.convnextv2 import convnextv2_att
from ultralytics.nn.models.fighting_model.backbone.repghost import RepGhostBottleneck
from ultralytics.nn.models.add_models.blockii import BiLevelRoutingAttention, AttentionLePE, Attention

from ultralytics.nn.models.fighting_model.conv.ODConv import ODConv2d
from ultralytics.nn.models.fighting_model.conv.MBConv import MBConvBlock
from ultralytics.nn.models.fighting_model.conv.CondConv import CondConv
from ultralytics.nn.models.fighting_model.conv.DynamicConv import DynamicConv
from ultralytics.nn.models.fighting_model.conv.HorNet import gnconv  # use: gnconv(dim), c1 = c2

__all__ = ('DoubleAttention', 'ACmix', 'AFT_FULL', 'AxialAttention', 'AxialImageTransformer', 'BAMBlock',
           'BiLevelRoutingAttention', 'CBAMBlock', 'CoAtNet', 'CoordAtt', 'CrissCrossAttention',
           'deformable_LKA_Attention_experimental', 'deformable_LKA_Attention', 'DAModule', 'DAT', 'LocalAttention',
           'ECAAttention', 'EMAU', 'EMSA',
           'EVCBlock', 'ExternalAttention', 'FocusedLinearAttention', 'GAMAttention', 'GFNet', 'HaloAttention', 'LKA',
           'MOATransformer', 'MobileViTAttention', 'MobileViTv2Attention',
           'MUSEAttention', 'OutlookAttention', 'ParNetAttention', 'PSA', 'ResidualAttention', 'S2Attention',
           'SEAttention', 'SpatialGroupEnhance',
           'ShuffleAttention', 'SimAM', 'SimplifiedScaledDotProductAttention', 'SKAttention', 'TripletAttention',
           'UFOAttention', 'WeightedPermuteMLP')
