
from fastapi import FastAPI, UploadFile, File
from requests.utils import requote_uri
import uvicorn                  # локальный host
import nest_asyncio             # асинхронные функций
import json
from io import BytesIO          # чтение файла в байтовом режиме
from PIL import Image           # работа с изображениями
from ultralytics import YOLO    # работа с моделью

nest_asyncio.apply() # включить асинхронные функций
app = FastAPI()      # создать экземпляр FastAPI

@app.post("/uploaded_images")                             # команда для post-запроса
async def uploaded_images(file: UploadFile = File(...)):  # функция для получения файла из post-запроса
    result = process_image(file)
    return {"result": result}                             # возвращает результат работы модели

@app.get("/")
def read_root():
    return {"Hello": "World"}

def process_image(image_file):
    image = Image.open(BytesIO(image_file.file.read()))
    processed_image = image.convert("RGB")
    model = YOLO("saved_weights/train14_3cl_340ep.pt")    # yolov8n.pt
    results = model.predict(source=processed_image, classes=[0,2], conf=0.75)  
    man = 0
    gun = 0
    for res in results:
        boxes = res.boxes
        for box in boxes:
            print(box)                                    # вывод всей информации работы модели в текст
            if int(box.cls[0]) == 0:
                man += 1
            elif int(box.cls[0]) == 1:
                gun += 1
    response = {"man": man, "gun": gun }
    
    return {"status": "success", "response": json.dumps(response)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  # запуск веб-приложений через локальный host
