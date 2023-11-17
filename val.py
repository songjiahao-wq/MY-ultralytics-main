from ultralytics import YOLO
if __name__ == '__main__':
    # model = YOLO("yolov8s.yaml")  # 从头开始构建新模型
    model = YOLO(r"D:\yanyi\project_process\ultralytics\runs\detect\train2\weights/best.pt")  # 加载预训练模型（推荐用于训练）
    results = model.val(data="NEU.yaml",batch=1)  # 在验证集上评估模型性能
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

#CLI
# yolo detect val model=yolov8n.pt  # val official model
# yolo detect val model=path/to/best.pt  # val custom model

#参数列表
# data	None	数据文件的路径，即 coco128.yaml
# imgsz	640	图像大小为标量或（H，W）列表，即（640，480）
# batch	16	每批图像数（自动批处理为 -1）
# save_json	False	将结果保存到 JSON 文件
# save_hybrid	False	保存混合版本的标签（标签 + 其他预测）
# conf	0.001	用于检测的对象置信度阈值
# iou	0.6	NMS 的联合 （IoU） 阈值上的交集
# max_det	300	每个图像的最大检测数
# half	True	使用半精度 （FP16）
# device	None	要运行的设备，即 CUDA 设备=0/1/2/3 或设备=CPU。
# dnn	False	使用 OpenCV DNN 进行 ONNX 推理
# plots	False	在训练期间显示绘图
# rect	False	矩形 val，每批都经过整理，以实现最小的填充
# split	val	数据集拆分用于验证，即“val”、“test”或“train”