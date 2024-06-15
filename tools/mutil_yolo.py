from ultralytics import YOLO
import torch
import os
from pathlib import Path
import logging
import traceback
# Change the working directory to 'run/'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ == '__main__':

    config_directory = r"ultralytics/cfg/models/config/attention"
    config_files = os.listdir(config_directory)
    for config_file in config_files:
        if config_file.endswith('.yaml'):
            config_path = os.path.join(config_directory, config_file)
            config_path = config_path.replace('yolov8-', 'yolov8s-')
            print(f"Training model with config: {config_path}")
            try:
                model = YOLO(config_path)  # 从头开始构建新模型
            except BaseException as e:
                # 打印出错误类型和错误位置
                traceback.print_exc()
                continue

