import os
import cv2
from ultralytics import YOLO

# ------------------------------
# Пути к модели и папкам
# ------------------------------
model_path = "runs/hagrid_yolo_improved2/weights/best.pt"
model = YOLO(model_path)

# Папки с новыми данными
image_input_dir = "new_data/images"
video_input_dir = "new_data/videos"

# Папки для сохранения результатов
image_output_dir = "runs/inference/images"
video_output_dir = "runs/inference/videos"
video_frames_dir = "runs/inference/video_frames"  
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(video_frames_dir, exist_ok=True)

# ------------------------------
# 1. Детекция на одном изображении
# ------------------------------
def infer_image(image_path):
    filename = os.path.basename(image_path)
    save_path = os.path.join(image_output_dir, filename)

    results = model(image_path, conf=0.25)
    annotated = results[0].plot()
    cv2.imwrite(save_path, annotated)
    print(f"Детекция выполнена → {save_path}")

# ------------------------------
# 2. Детекция на папке изображений
# ------------------------------
def infer_image_folder(folder_path):
    for fn in os.listdir(folder_path):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            infer_image(os.path.join(folder_path, fn))

# ------------------------------
# 3. Детекция на видео
# ------------------------------
def infer_video(video_path):
    cap = cv2.VideoCapture(video_path)
    name = os.path.basename(video_path).split(".")[0]

    # Папка для кадров видео
    frame_folder = os.path.join(video_frames_dir, name)
    os.makedirs(frame_folder, exist_ok=True)

    save_video_path = os.path.join(video_output_dir, f"{name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(save_video_path, fourcc, fps, (w, h))

    print(f"Обработка видео: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)
        annotated = results[0].plot()

        # Сохраняем аннотированный кадр
        frame_path = os.path.join(frame_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, annotated)

        # Записываем в итоговое видео
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Видео сохранено → {save_video_path}")
    print(f"Кадры сохранены → {frame_folder}")

# ------------------------------
# 4. Автоматический запуск
# ------------------------------
if __name__ == "__main__":
    # Обработка всех изображений
    if os.path.exists(image_input_dir):
        infer_image_folder(image_input_dir)
    else:
        print(f"Папка с изображениями не найдена: {image_input_dir}")

    # Обработка всех видео
    video_extensions = (".mp4", ".avi", ".mov", ".webm")
    if os.path.exists(video_input_dir):
        for fn in os.listdir(video_input_dir):
            if fn.lower().endswith(video_extensions):
                infer_video(os.path.join(video_input_dir, fn))
    else:
        print(f"Папка с видео не найдена: {video_input_dir}")
