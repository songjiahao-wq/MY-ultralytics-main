from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/models/v8/yolov8s-pose.yaml")  # build a new model from scratch
model = YOLO("YOLOv8s-pose.pt")  # load a pretrained model (recommended for training)


path = model.export(format="onnx")  # export the model to ONNX format