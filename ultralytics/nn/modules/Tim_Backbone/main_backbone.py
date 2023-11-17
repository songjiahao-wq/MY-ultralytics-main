import torch, timm
from thop import clever_format, profile
import sys
print("可以导入的网络列表", timm.list_models())
print("可以导入的带预训练模型的网络列表", timm.list_models())

# ## 使用通配符列出可用的不同ResNet变体
# resnet_model_list = timm.list_models("*resnet*")
# pretrain_resnet_model_list = timm.list_models("*resnet*", pretrained=False)
# print(resnet_model_list)
# print(pretrain_resnet_model_list)
#
# # num_classes=10改变输出类别数
# # in_chans=3改变输入通道数
# model_cspresnet50_out10 = timm.create_model("cspresnet50", pretrained=False, num_classes=10, in_chans=3)
# print(model_cspresnet50_out10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 3, 640, 640).to(device)

# model = timm.create_model('edgenext_small', pretrained=False, features_only=True)
model = timm.create_model('adv_inception_v3', pretrained=False, features_only=True)
model.to(device)
model.eval()
print(model.feature_info.channels())
for feature in model(dummy_input,):
    print(feature.size())

flops, params = profile(model.to(device), (dummy_input,), verbose=False)
flops, params = clever_format([flops * 2, params], "%.3f")
print('Total FLOPS: %s' % (flops))
print('Total params: %s' % (params))

m = timm.create_model('ecaresnet101d', features_only=True, output_stride=32, out_indices=(0, 1,2,3, 4), pretrained=False)
print(f'Feature channels: {m.feature_info.channels()}')
print(f'Feature reduction: {m.feature_info.reduction()}')
o = m(torch.randn(1, 3, 640, 640))
for x in o:
    print(x.shape)