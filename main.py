import os
import time
import datetime

import torch
import cv2 as cv

from config import settings
from utils import show_images, draw_bounding_boxes
from tracking import simple_online_realtime_tracking


def main():
    print(f'Image Folder: {settings.IMAGE_FOLDER}')

    image_names = sorted(
        [f for f in os.listdir(settings.IMAGE_FOLDER) if os.path.isfile(os.path.join(settings.IMAGE_FOLDER, f))])

    simple_online_realtime_tracking(image_names=image_names, image_folder=settings.IMAGE_FOLDER)
    dummy = -32


if __name__ == "__main__":
    print(cv.__version__)
    print(torch.__version__)

    if torch.cuda.is_available():
        print(f'CUDA Current Device: {torch.cuda.current_device()}')
    else:
        raise RuntimeError('No GPU found!')

    print(f'{settings.APPLICATION_NAME} started at {datetime.datetime.now()}')
    time1 = time.time()
    main()
    time2 = time.time()
    print(f'{settings.APPLICATION_NAME} finished at {datetime.datetime.now()}')
    print(f'{settings.APPLICATION_NAME} took {time2 - time1} seconds')
