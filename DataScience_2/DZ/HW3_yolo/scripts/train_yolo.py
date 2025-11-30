from ultralytics import YOLO

# Путь к весам и датасету
weights_path = "weights/yolov8s.pt"
data_path = "data/data.yaml"

model = YOLO(weights_path)

model.train(
    data=data_path,
    epochs=20,
    batch=2,
    imgsz=320,
    name="hagrid_yolo_improved",
    project="runs",
    device="cpu",
    workers=0,
    verbose=True
)

