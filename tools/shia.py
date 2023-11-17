import supervision as sv
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("./yolov8m.pt")

def callback(x: np.ndarray) -> sv.Detections:
    result = model(x, verbose=False, conf=0.25)[0]
    return sv.Detections.from_ultralytics(result)
image = cv2.imread("./1.png")

slicer = sv.InferenceSlicer(callback=callback)
sliced_detections = slicer(image=image)
box_annotator = sv.BoxAnnotator()

sliced_image = box_annotator.annotate(image.copy(), detections=sliced_detections)
cv2.imwrite('./out.png',sliced_image)
sv.plot_image(sliced_image)