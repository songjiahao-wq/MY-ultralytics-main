from ultralytics import YOLO

# Load a model
model = YOLO(r"D:\my_job\MY_Github\MY-ultralytics-main\ultralytics\cfg\models\v8\yolov8s.yaml")  # build a new model from scratch
model = YOLO("YOLOv8s.pt")  # load a pretrained model (recommended for training)


path = model.export(format="engine")  # export the model to ONNX format