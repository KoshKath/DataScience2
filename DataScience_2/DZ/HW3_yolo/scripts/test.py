from PIL import Image
import os

image_dir = "data/train/images"  # путь к твоим тренировочным изображениям
for img_name in os.listdir(image_dir):
    if img_name.endswith((".jpg", ".png")):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path)
        print(img_name, img.size)  # выводит (ширина, высота)
