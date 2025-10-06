from ultralytics import YOLO
import torch
import os
from PIL import Image, ImageDraw, ImageFilter


def load_yolo_model():
    """
    Returns yolo model with pretrained weights
    """
    model = YOLO("../api/weights/yolo_face_detection.pt")
    model.eval()
    return model


def blur_detections(image: Image.Image, results):
    """
    Накладывает блюр на детекции YOLO (совместимо с YOLOv8 и YOLOv11).
    """
    # Для надёжности берём первый элемент, если results — список
    if isinstance(results, list):
        results = results[0]

    # YOLOv11: boxes -> ultralytics.engine.results.Boxes
    if results.boxes is None or len(results.boxes) == 0:
        return image

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        region = image.crop((x1, y1, x2, y2))
        blurred_region = region.filter(ImageFilter.GaussianBlur(25))
        image.paste(blurred_region, (x1, y1, x2, y2))

    return image


def load_lstm_model():
    """
    Returns custom model based on LSTM with word2vec embedding and Bahdanau Attention
    """
    pass
