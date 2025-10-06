import logging
from contextlib import asynccontextmanager

from typing import List
import PIL

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import base64
import io
from pydantic import BaseModel, Field
from utils.model_func import load_yolo_model, load_lstm_model, blur_detections


logger = logging.getLogger("uvicorn.info")


# Определение класса запроса для классификации текста
class TextInput(BaseModel):
    text: str  # Текст, введенный пользователем для классификации


# Определение класса ответа для классификации текста
class TextResponse(BaseModel):
    label: str  # метка класса (positive/negative)
    prob: float  # вероятность, связанная с меткой


# Определение класса ответа для детекции одного оъекта на изображении
# class Detection(BaseModel):
#     """
#     Описывает один обнаруженный объект на изображении.
#     """

#     box: List[int] = Field(
#         ..., description="Координаты рамки в формате [x1, y1, x2, y2]"
#     )
#     label: str = Field(..., description="Метка класса объекта")
#     confidence: float = Field(..., description="Уверенность модели в детекции")


# Определение класса ответа для детекции всех лиц на изображении
class ImageResponse(BaseModel):
    faces_count: int = Field(..., description="Количество обнаруженных лиц")
    blurred_image_base64: str = Field(..., description="Изображение с блюром в формате base64")


yolo_model = None  # глобальная переменная для yolo модели
lstm_model = None  # глобальная переменная для кастомной lstm модели


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """
    global yolo_model
    global lstm_model
    # Загрузка yolo модели
    yolo_model = load_yolo_model()
    logger.info("YOLO model loaded")
    # Загрузка кастомной lstm модели
    lstm_model = load_lstm_model()
    logger.info("LSTM model loaded")
    yield
    # Удаление моделей и освобождение ресурсов
    del yolo_model, lstm_model


app = FastAPI(lifespan=lifespan)


@app.get("/")
def return_info():
    """
    Возвращает приветственное сообщение при обращении к корневому маршруту API.
    """
    return "Hello FastAPI!"


@app.post("/dtctn_image")
def detection_image(file: UploadFile):
    """
    Endpoint для детекции лиц на изображении.
    Принимает файл изображения, обрабатывает его, делает детекцию и возвращает изображение с bounding-box  в виде заблюриной маски.
    """
    # Открытие изображения
    image = PIL.Image.open(file.file).convert("RGB")
    # Детекция лиц на изображении
    results = yolo_model(image)
    if isinstance(results, list):
        results = results[0]
    # Подсчет количества детекций
    faces_count = 0
    if results.boxes is not None:
        faces_count = len(results.boxes)
    # Получение маски
    blurred = blur_detections(image.copy(), results)
    # Сохраняем в память как байты JPEG
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG")
    # buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    # Формирование ответа
    response = ImageResponse(
        faces_count=faces_count,
        blurred_image_base64=img_base64
    )
    return response


if __name__ == "__main__":
    # Запуск приложения на localhost с использованием Uvicorn
    # производится из командной строки: python your/path/api/main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
