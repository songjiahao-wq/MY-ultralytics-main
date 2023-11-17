# 训练模型
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import glob
import os
if __name__ == '__main__':
    model = YOLO("./yolov8n.pt")
    root_dir = r'D:\project\7.4\MY-YOLOv8\ultralytics\assets'  # 替换为你的ultralytics目录的路径
    # 遍历所有 .jpg, .jpeg, .png 等照片格式文件
    image_types = ('*.jpeg', '*.jpg', '*.png', '*.bmp')
    all_images = []
    for ext in image_types:
        all_images.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))

    path = r"D:\project\7.4\MY-YOLOv8\crop_runs"
    if not os.path.exists(path):
        os.mkdir(path)
    for index_i,img_name in enumerate(all_images):
        img = cv2.imread(img_name)
        results = model.predict(source=img_name, classes=[0], conf=0.5, save=True)
        # 获取图像名和其父目录名
        base_name = os.path.basename(img_name).rsplit('.', 1)[0]
        parent_name = os.path.basename(os.path.dirname(img_name))
        # Process results list
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Boxes object for bbox outputs
            for i, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                crop = img[r[1]:r[3], r[0]:r[2]]
                # 按照父目录文件名-名字编号1-bbox编号命名并保存
                new_filename = os.path.join(path, f"{parent_name}-{base_name}-{index_i}-{i}.jpg")

                cv2.imwrite(new_filename, crop)

