import os

from scipy.optimize import linear_sum_assignment
import numpy as np
from ultralytics import YOLO
import cv2 as cv

from metric import total_cost


class Obstacle:
    def __init__(self, idx, box, age=1, unmatched_age=0):
        self.idx = idx
        self.box = box
        self.age = age
        self.unmatched_age = unmatched_age


def associate(old_boxes, new_boxes):
    """
    old_boxes will represent the former bounding boxes (at time 0)
    new_boxes will represent the new bounding boxes (at time 1)
    Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
    """
    if (len(new_boxes) == 0) and (len(old_boxes) == 0):
        return [], [], []
    elif len(old_boxes) == 0:
        return [], new_boxes, []
    elif len(new_boxes) == 0:
        return [], [], old_boxes

    # Define a new IOU Matrix nxm with old and new boxes
    iou_matrix = np.zeros(shape=(len(old_boxes), len(new_boxes)), dtype=np.float32)

    # Go through boxes and store the IOU value for each box
    # You can also use the more challenging cost but still use IOU as a reference for convenience (use as a filter only)
    for i, old_box in enumerate(old_boxes):
        for j, new_box in enumerate(new_boxes):
            iou_matrix[i][j] = total_cost(old_box, new_box)

    # Call for the Hungarian Algorithm
    hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
    hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

    # Create new unmatched lists for old and new boxes
    matches, unmatched_detections, unmatched_trackers = [], [], []

    # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched
    # Else: add the match
    for h in hungarian_matrix:
        if iou_matrix[h[0], h[1]] < 0.3:
            unmatched_trackers.append(old_boxes[h[0]])
            unmatched_detections.append(new_boxes[h[1]])
        else:
            matches.append(h.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty(shape=(0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # Go through old boxes, if no matched detection, add it to the unmatched_old_boxes
    for t, trk in enumerate(old_boxes):
        if t not in hungarian_matrix[:, 0]:
            unmatched_trackers.append(trk)

    # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
    for d, det in enumerate(new_boxes):
        if d not in hungarian_matrix[:, 1]:
            unmatched_detections.append(det)

    return matches, unmatched_detections, unmatched_trackers


def simple_online_realtime_tracking(image_names: list[str], image_folder: str):
    model = YOLO('yolov8n.pt')

    for idx, image_name in enumerate(image_names):
        image = cv.imread(os.path.join(image_folder, image_name))
        results = model(image)
        boxes = results[0].boxes
        names_dict = results[0].names

        b_boxes = [box.xyxy.cpu().reshape(4, ).numpy() for box in boxes]
        class_names = [names_dict[int(box.cls.cpu()[0])] for box in boxes]

        if idx == 0:
            continue

        dummy = -32
