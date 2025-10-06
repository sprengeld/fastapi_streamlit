from ultralytics import YOLO
import torch
import os
from PIL import Image, ImageDraw, ImageFilter
from utils.LSTM_model import LSTMWord2VecBahdanauAttention
from utils.prepare_embedding import preprocess_single_string, load_vocab


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
    model = LSTMWord2VecBahdanauAttention()
    model.load_state_dict(torch.load("../api/weights/LSTM_w2v_epoch_5.pth"))
    model.eval()
    return model


SEQ_LEN = 64
vocab_to_int = load_vocab("../api/weights/vocab.pkl")

def text_to_indices(text: str):
    # препроцессинг текста
    prep_text = preprocess_single_string(text, SEQ_LEN, vocab_to_int)
    # Приводим к типу Long для embedding
    prep_text = prep_text.long()
    # Добавляем batch dimension: (1, seq_len)
    prep_text = prep_text.unsqueeze(0)
    return prep_text


