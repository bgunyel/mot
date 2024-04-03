def box_iou(box1, box2):
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


def total_cost(old_box, new_box, iou_thresh=0.3, linear_thresh=10000, exp_thresh=0.5):
    cost = box_iou(box1=old_box, box2=new_box)
    return cost
