from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    # results = model(['ultralytics/assets/zidane.jpg'])  # return a list of Results objects
    results = model.predict(source='ultralytics/assets/zidane.jpg', conf=0.5, save=True)
