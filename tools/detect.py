# 训练模型
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

if __name__ == '__main__':
    model = YOLO("./yolov8n.pt")
    pred_id=0
    if pred_id==0:
        # 预测模型
        results = model.predict(source="ultralytics/assets/zidane.jpg", classes=[0], conf=0.5, save=True)[0]  # 0是摄像头，详细参数见3.2
        # boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
        resize_img = cv2.imread('ultralytics/assets/zidane.jpg') # 待裁剪照片
        w, h = resize_img.shape[0], resize_img.shape[1]
        # print(boxes)
        boxes = results.boxes  # Boxes object for bbox outputs[1], box[2], box[3]
        print(boxes.xyxyn)
        for box in np.array(boxes.xyxy.cpu()):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            print(x1, y1, x2, y2)
            resized_img = resize_img[y1:y2, x1:x2]
            cv2.imshow('a', resized_img)
            cv2.waitKey(0)
    elif pred_id==1:
        dir_path = Path("ultralytics/")
        folder_name = dir_path.parent.name
        files = list(dir_path.glob("*.jpg"))
        for i, img_name in enumerate(files):
            name = img_name.stem
            results = model(img_name)[0]  # return a list of Results objects
            boxes = results.boxes  # Boxes object for bbox outputs[1], box[2], box[3]
            resize_img = cv2.imread('ultralytics/assets/zidane.jpg')  # 待裁剪照片
            for j,box in enumerate(np.array(boxes.xyxy.cpu())):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                print(x1, y1, x2, y2)
                resized_img = resize_img[y1:y2, x1:x2]
                new_filename = "{}-{}-{}.jpg".format(folder_name, i + 1,j)
                cv2.imwrite('crop/'+new_filename,resized_img)
                cv2.imshow('a', resized_img)
                cv2.waitKey(0)
"""
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('bus.jpg', save=True, imgsz=320, conf=0.5)
"""