from datasets import load_dataset
import os
import random
from tqdm import tqdm

# -------------------------------
# Настройки
# -------------------------------
OUTPUT_DIR = "../data/hagrid_yolo"
TRAIN_RATIO = 0.8

# Оригинальные ID HaGRID для выбранных жестов
TARGET_LABELS = [2, 5, 7]   # one, fist, mute
CLASS_NAMES = ["fist", "mute", "one"]  # соответствие YOLO ID после перенумерации

# Создаем папки
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

print("Загружаем датасет (без streaming)...")
ds = load_dataset("cj-mills/hagrid-sample-30k-384p", split="train")

# -------------------------------
# Фильтрация по выбранным классам с прогресс-баром
# -------------------------------
print("Фильтруем только нужные классы...")

filtered_examples = []
for ex in tqdm(ds, desc="Фильтрация", unit="img"):
    if ex["label"] in TARGET_LABELS:
        filtered_examples.append(ex)

print("Всего найдено подходящих изображений:", len(filtered_examples))

# Перемешиваем
random.shuffle(filtered_examples)

# -------------------------------
# Разбивка на train / val
# -------------------------------
train_count = int(len(filtered_examples) * TRAIN_RATIO)
train_ds = filtered_examples[:train_count]
val_ds = filtered_examples[train_count:]

# -------------------------------
# Сохранение изображений и YOLO labels
# -------------------------------
def save_split(split_ds, split_name):
    print(f"\nСохраняем {split_name}...")
    for idx, ex in enumerate(tqdm(split_ds, desc=f"{split_name}", unit="img")):
        label = ex["label"]
        # перенумерация ID: из HaGRID ID → 0..2 для YOLO
        yolo_class = TARGET_LABELS.index(label)

        # Сохраняем изображение
        img_path = f"{OUTPUT_DIR}/{split_name}/images/{idx}.jpg"
        ex["image"].save(img_path)

        # Сохраняем YOLO label (bbox = весь кадр)
        label_path = f"{OUTPUT_DIR}/{split_name}/labels/{idx}.txt"
        with open(label_path, "w") as f:
            f.write(f"{yolo_class} 0.5 0.5 1.0 1.0\n")

save_split(train_ds, "train")
save_split(val_ds, "val")

# -------------------------------
# Создание data.yaml
# -------------------------------
yaml_path = f"{OUTPUT_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {OUTPUT_DIR}/train/images\n")
    f.write(f"val: {OUTPUT_DIR}/val/images\n")
    f.write(f"nc: {len(CLASS_NAMES)}\n")
    f.write("names:\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {i}: {name}\n")

print("\nГотово! data.yaml создан:", yaml_path)
