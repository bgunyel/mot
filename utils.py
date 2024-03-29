import glob
import os
import cv2 as cv


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
