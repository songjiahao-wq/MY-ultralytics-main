import os
import subprocess
from ultralytics import YOLO
from pathlib import Path

#运行的时候需要修改yolo里Detect/foward的输出
# Change the working directory to 'run/'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_directory = r"/home/lvting/A_lvting/MY-YOLOv8/ultralytics/models/config/improve/"
config_files = os.listdir(config_directory)

for config_file in config_files:
    if config_file.endswith('.yaml'):
        config_path = os.path.join(config_directory, config_file)
        config_path = config_path.replace('yolov8-', 'yolov8n-')
        print(f"Training model with config: {config_path}")
        try:
          model = YOLO(config_path)  # 从头开始构建新模型
          results = model.train(data="ultralytics/datasets/my_data.yaml",
                          epochs=150, batch=16, imgsz=640, workers=8, name=Path(model.cfg).stem)  # 训练模型
        except:
          continue
        #subprocess.run(['yolo', 'detect', 'train', 'data=ultralytics/datasets/my_data.yaml.yaml', 'model=', config_path,'epochs=150' ,'imgsz=640','batch=16','workers=8','name=', Path(config_path).stem])

# yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640