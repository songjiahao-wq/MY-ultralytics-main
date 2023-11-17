from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':


    # 加载模型
    model = YOLO("ultralytics/models/config/yolov8s.yaml")  # 从头开始构建新模型
    # model = YOLO("runs/detect/v5n9/weights/best.pt")  # 加载预训练模型（推荐用于训练）
    # Use the model
    results = model.train(data="ultralytics/datasets/my_data.yaml",
                          epochs=100, batch=16, imgsz=640, workers=4, name=Path(model.cfg).stem)  # 训练模型
    # results = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    # """ 断点续练"""
    # model = YOLO(r"F:\sjh\code\ultralytics\runs\detect\cfg\weights\last.pt")
    # model.train(resume=True)

# CIL调用方式
# # Build a new model from YAML and start training from scratch
# yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

# # Start training from a pretrained *.pt model
# yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

# # Build a new model from YAML, transfer pretrained weights to it and start training
# yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640


#参数列表
# model	None	模型文件的路径，即 yolov8n.pt，yolov8n.yaml
# data	None	数据文件的路径，即 coco128.yaml
# epochs	100	要训练的时期数
# patience	50	等待没有明显改善以提前停止训练的时期
# batch	16	每批图像数（自动批处理为 -1）
# imgsz	640	输入图像的大小为整数或 W，H
# save	True	保存列车检查点并预测结果
# save_period	-1	每隔 x 个纪元保存检查点（如果< 1 则禁用）
# cache	False	真/内存、磁盘或假。使用缓存加载数据
# device	None	要运行的设备，即 CUDA 设备=0 或设备=0，1，2，3 或设备=CPU
# workers	8	用于数据加载的工作线程数（如果为 DDP，则按 RANK 计算）
# project	None	项目名称
# name	None	实验名称
# exist_ok	False	是否覆盖现有实验
# pretrained	False	是否使用预训练模型
# optimizer	'SGD'	要使用的优化器， choices=['SGD'， 'Adam'， 'AdamW'， 'RMSProp']
# verbose	False	是否打印详细输出
# seed	0	用于重现性的随机种子
# deterministic	True	是否启用确定性模式
# single_cls	False	将多类数据训练为单类
# rect	False	矩形训练，每批整理以获得最小的填充
# cos_lr	False	使用余弦学习速率调度程序
# close_mosaic	0	（int） 禁用最终纪元的镶嵌增强
# resume	False	从上一个检查点恢复训练
# amp	True	自动混合精度 （AMP） 训练，选择=[真，假]
# fraction	1.0	要训练的数据集分数（默认值为 1.0，训练集中的所有图像）
# profile	False	在记录器训练期间分析 ONNX 和 TensorRT 速度
# lr0	0.01	初始学习率（即SGD=1E-2，亚当=1E-3）
# lrf	0.01	最终学习率 （LR0 * LRF）
# momentum	0.937	新加坡元动量/亚当贝塔1
# weight_decay	0.0005	优化器重量衰减 5E-4
# warmup_epochs	3.0	热身时期（分数确定）
# warmup_momentum	0.8	预热初始动量
# warmup_bias_lr	0.1	预热初始偏置 LR
# box	7.5	箱子损失收益
# cls	0.5	CLS 损耗增益（随像素缩放）
# dfl	1.5	DFL 损失收益
# pose	12.0	姿势损失增益（仅姿势）
# kobj	2.0	关键点 OBJ 损失增益（仅姿势）
# label_smoothing	0.0	标签平滑（分数）
# nbs	64	公称批量大小
# overlap_mask	True	训练期间口罩应重叠（仅限分段训练）
# mask_ratio	4	模板下采样率（仅限段序列）
# dropout	0.0	使用 Dropout 正则化（仅限分类训练）
# val	True	在训练期间验证/测试