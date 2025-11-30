import os
import shutil
import json
import cv2
import numpy as np
from ultralytics import YOLO


# -------------------------------------------------
#                НАСТРОЙКИ ПУТЕЙ
# -------------------------------------------------
WEIGHTS_PATH = "runs/hagrid_yolo_improved2/weights/best.pt"
DATA_YAML = "data/data.yaml"
VAL_IMAGES = "data/val/images"
VAL_LABELS = "data/val/labels"

OUTPUT_ROOT = "runs/hagrid_yolo_improved2/val/analysis"


# -------------------------------------------------
#         Чистим предыдущие результаты
# -------------------------------------------------
if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

folders = {
    "annotated": "annotated",
    "false_positives": "false_positives",
    "false_negatives": "false_negatives",
    "correct_predictions": "correct_predictions",
    "misclassified": "misclassified"
}

for folder in folders.values():
    os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)


# -------------------------------------------------
#           Загружаем модель
# -------------------------------------------------
print(f"\nЗагрузка модели: {WEIGHTS_PATH}")
model = YOLO(WEIGHTS_PATH)


# -------------------------------------------------
#             Запускаем валидацию
# -------------------------------------------------
print("\nЗапуск валидации YOLO:")

results = model.val(
    data=DATA_YAML,
    split="val",
    imgsz=320,
    batch=2,
    device="cpu"
)

print("\nВалидация завершена.")


# -------------------------------------------------
#            Сохраняем метрики
# -------------------------------------------------
metrics = {
    "mAP50": float(results.box.map50),
    "mAP50-95": float(results.box.map),
    "precision": results.box.p.tolist(),
    "recall": results.box.r.tolist(),
    "f1_score": results.box.f1.tolist(),
    "classes": results.names,
    "confusion_matrix": results.confusion_matrix.matrix.astype(float).tolist()
}

with open(os.path.join(OUTPUT_ROOT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("Метрики сохранены.\n")


# -------------------------------------------------
#   Функция загрузки ground truth разметки
# -------------------------------------------------
def load_gt_labels(path):
    if not os.path.exists(path):
        return []
    data = []
    with open(path, "r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            data.append((int(cls), x, y, w, h))
    return data


# -------------------------------------------------
#           IoU для xywh-формата
# -------------------------------------------------
def iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    b1 = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
    b2 = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]

    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# -------------------------------------------------
#       Анализ каждого изображения вручную
# -------------------------------------------------
print("\nАнализ изображения:")

for filename in os.listdir(VAL_IMAGES):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(VAL_IMAGES, filename)
    lbl_path = os.path.join(VAL_LABELS, filename.rsplit(".", 1)[0] + ".txt")

    gt_boxes = load_gt_labels(lbl_path)

    pred = model(img_path, conf=0.25)[0]
    annotated = pred.plot()

    # сохраняем аннотированное
    cv2.imwrite(os.path.join(OUTPUT_ROOT, "annotated", filename), annotated)

    preds = pred.boxes
    pred_cls = preds.cls.cpu().numpy() if len(preds) else []
    pred_conf = preds.conf.cpu().numpy() if len(preds) else []
    pred_xyxy = preds.xyxy.cpu().numpy() if len(preds) else []

    # ситуация: модель ничего не нашла
    if len(preds) == 0 and len(gt_boxes) > 0:
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "false_negatives", filename), annotated)
        continue

    # ситуация: модель нашла, но GT нет
    if len(preds) > 0 and len(gt_boxes) == 0:
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "false_positives", filename), annotated)
        continue

    # сравнение классов
    classified = False

    for cls_pred in pred_cls:
        match = any(cls_pred == gt_cls for gt_cls, *_ in gt_boxes)
        wrong = any(cls_pred != gt_cls for gt_cls, *_ in gt_boxes)

        if match:
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "correct_predictions", filename), annotated)
            classified = True
            break
        if wrong:
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "misclassified", filename), annotated)
            classified = True
            break

    if not classified and len(gt_boxes) > 0:
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "false_negatives", filename), annotated)

print("\nАнализ сохранён в:")
print(OUTPUT_ROOT)
