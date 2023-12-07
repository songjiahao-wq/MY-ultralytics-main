from ultralytics import YOLO
import torch
if __name__ == '__main__':
    # 加载模型
    # Create model
    device = torch.device('cuda:0')
    im = torch.rand(1, 3, 640, 640).to(device)
    model = YOLO("ultralytics/cfg/models/config/attention/yolov8s-attention2-BAMBlock.yaml")  # 从头开始构建新模型
    # model(im, profile=True)
    #model.predict('ultralytics/assets', save=True, imgsz=320, conf=0.5)
    # model = YOLO("yolov8s.pt")  # 加载预训练模型（推荐用于训练）
    #yolo detect train resume model=last.pt
