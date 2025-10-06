import requests
import streamlit as st
import base64
from PIL import Image
import io

st.title("FastAPI application for face detection and ...")

tab1, tab2 = st.tabs(["Image", "Text"])


def main():
    with tab1:
        # загрузка изображения
        image = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

        if st.button("Запустить детекцию") and image is not None:
            # format data
            files = {"file": (image.name, image, "multipart/form-data")}
            # отправка данных на FastAPI и эндпоинт
            response = requests.post("http://127.0.0.1:8000/dtctn_image", files=files)

            # обработка ответа
            if response.status_code == 200:
                data = response.json()
                
                ## декодирование base64 в изображение
                img_bytes = base64.b64decode(data["blurred_image_base64"])
                res_img = Image.open(io.BytesIO(img_bytes))
                ## количество обнаруженных лиц
                faces = data["faces_count"]

                # вывод результатов
                st.image(res_img, caption="Результат детекции")
                st.write(f"Найдено лиц: {faces}")
            else:
                st.error(f"Ошибка API: {response.status_code}")

    with tab2:
        st.write("Classify your text")


if __name__ == "__main__":
    main()
