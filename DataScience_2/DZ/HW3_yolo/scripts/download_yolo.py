from ultralytics import YOLO
import os

weights_dir = "../weights"
os.makedirs(weights_dir, exist_ok=True)

model = YOLO("yolov8s.pt")  # small версия YOLOv8
model.save(os.path.join(weights_dir, "yolov8s.pt"))

print("YOLOv8s вес сохранён в:", os.path.join(weights_dir, "yolov8s.pt"))
