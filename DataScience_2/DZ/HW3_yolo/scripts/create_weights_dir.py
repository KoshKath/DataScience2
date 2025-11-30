import os

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
print(f"Папка для весов создана или уже существует: {WEIGHTS_DIR}")
