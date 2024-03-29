import os

import torch
import cv2 as cv
from ultralytics import YOLO

from config import settings
from utils import show_images


def manin(name: str):
    print(name)
    print(settings.IMAGE_FOLDER)

    image_names = sorted([f for f in os.listdir(settings.IMAGE_FOLDER) if os.path.isfile(os.path.join(settings.IMAGE_FOLDER, f))])

    model = YOLO('yolov8n.pt')

    for image_name in image_names:
        image = cv.imread(os.path.join(settings.IMAGE_FOLDER, image_name))
        results = model(image)
        boxes = results[0].boxes

        for box in boxes.xyxy:
            cv.rectangle(
                img=image,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])),
                color=(0, 255, 0),
                thickness=2
            )
        cv.imwrite(filename=os.path.join(settings.OUT_FOLDER, image_name), img=image)
        dummy = -32





if __name__ == "__main__":
    print(cv.__version__)
    print(torch.__version__)

    if torch.cuda.is_available():
        print(f'CUDA Current Device: {torch.cuda.current_device()}')
    else:
        raise RuntimeError('No GPU found!')

    manin(name=settings.APPLICATION_NAME)
