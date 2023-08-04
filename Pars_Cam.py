
from fastapi import FastAPI, UploadFile, File
import nest_asyncio
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import cv2
import requests
import torch
import json, os

nest_asyncio.apply()

# Load the model
model = YOLO("saved_weights/train14_3cl_340ep.pt")
path_output_dir = 'gun/'

def process_image():
    ip_cam = 'rtsp://192.168.3.2:4747/video'
    image = cv2.VideoCapture(0) # (0) Веб-камера  \ ip_cam
    # Loop through the video frames
    count = 0
    while image.isOpened():
            # Read a frame from the video
            success, frame = image.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = model(frame)
                annotated_frame = results[0].plot()
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                for res in results:
                    boxes = res.boxes
                    for box in boxes:
                            #print(box)
                            if int(box.cls[0]) == 1:
                                pars_img = cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, frame) # Сохраняет картинку если обнаружен 1й класс "gan"
                                count += 1
                                pass
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
    # Extract information about number of guns and humans from results:
    image.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_image()