import os

import cv2 as cv
import numpy as np


def show_images(image_folder: str) -> None:
    image_names = sorted(
        [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    )

    window_name = f'{len(image_names)} images @ {image_folder}'
    cv.namedWindow(winname=window_name, flags=cv.WINDOW_AUTOSIZE)

    for image_name in image_names:
        image = cv.imread(os.path.join(image_folder, image_name))
        cv.imshow(winname=window_name, mat=image)
        cv.waitKey()

    cv.destroyAllWindows()


def id_to_color(idx):
    """
    Random function to convert an id to a color
    """
    blue = idx * 5 % 256
    green = idx * 12 % 256
    red = idx * 23 % 256
    return red, green, blue


def draw_bounding_boxes(
        image: np.ndarray,
        boxes: list[np.ndarray],
        names: list[str],
        colors: list[tuple[int, int, int]]
) -> np.ndarray:

    for idx, box in enumerate(boxes):
        cv.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=colors[idx],
            thickness=2
        )
        cv.putText(
            img=image,
            text=names[idx],
            org=(int(box[0]), int(box[1])),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=colors[idx],
            thickness=2
        )

    return image
