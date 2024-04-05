from math import sqrt, exp

from utils import convert_box_xywh_to_xyxy


def check_division_by_0(value, epsilon=0.01):
    return value if value > epsilon else epsilon


def box_iou(box1, box2):
    """
        :param box1: represented in (x, y, w, h) format
        :param box2: represented in (x, y, w, h) format
        :return:
        """
    box1 = convert_box_xywh_to_xyxy(box=box1)
    box2 = convert_box_xywh_to_xyxy(box=box2)

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = (box1_area + box2_area) - intersection
    iou = intersection / float(union)
    return iou


def sanchez_matilla(box1, box2, width=1280, height=360):
    """
    :param box1: represented in (x, y, w, h) format
    :param box2: represented in (x, y, w, h) format
    :param width: image width
    :param height: image height
    :return:
    """
    Q_dist = sqrt(pow(width, 2) + pow(height, 2))
    Q_shape = width * height
    distance_term = Q_dist / check_division_by_0(sqrt(pow(box1[0] - box2[0], 2) + pow(box1[1] - box2[1], 2)))
    shape_term = Q_shape / check_division_by_0(sqrt(pow(box1[2] - box2[2], 2) + pow(box1[3] - box2[3], 2)))
    linear_similarity = distance_term * shape_term
    return linear_similarity


def yu(box1, box2):
    """
        :param box1: represented in (x, y, w, h) format
        :param box2: represented in (x, y, w, h) format
        :return:
        """
    w1 = -0.5
    w2 = -1.5
    a = (box1[0] - box2[0]) / check_division_by_0(box1[2])
    b = (box1[1] - box2[1]) / check_division_by_0(box1[3])
    c = abs(box1[3] - box2[3]) / (box1[3] + box2[3])
    d = abs(box1[2] - box2[2]) / (box1[2] + box2[2])
    ab = (a * a + b * b) * w1
    cd = (c + d) * w2
    exponential_similarity = exp(ab) * exp(cd)
    return exponential_similarity


def total_similarity(old_box, new_box, iou_thresh=0.3, linear_thresh=10000, exp_thresh=0.5):
    iou_score = box_iou(old_box, new_box)
    linear_similarity = sanchez_matilla(old_box, new_box)
    exponential_similarity = yu(old_box, new_box)

    similarity = 0
    if iou_score >= iou_thresh and linear_similarity >= linear_thresh and exponential_similarity >= exp_thresh:
        similarity = iou_score

    return similarity
