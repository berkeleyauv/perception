import csv
import numpy as np
from math import sqrt
from shapely.geometry import Polygon
from perception.vis.FrameWrapper import FrameWrapper
from perception.tasks.gate.GateSegmentationAlgoA import GateSegmentationAlgoA
from perception.tasks.dice.DiceDetector import DiceDetector
from perception.tasks.ReturnStructs import DiceBoxes
import os

# TODO: HC = Hardcoded

def test_segmentation():
    data_sources = ["../perception/vis/datasets/dice.mp4"] # HC
    gt_filename = "labels/DiceLabels.csv" # HC
    best_metric = 0.01
    returnedmetric = run_algo(data_sources, DiceDetector(), gt_filename)
    print('FINAL METRIC: ', returnedmetric)
    assert returnedmetric >= best_metric # HC

def run_algo(data_source, algorithm, gt_filename):
    data = FrameWrapper(data_source, 0.25)
    comp_data = []

    gt_data = read_csv(gt_filename)

    for frame in data:
        contours = algorithm.analyze(frame, debug=False, slider_vals={
            'heuristic_threshold': 35,
            'run_both': False,
            'centroid_distance_weight': 1,
            'area_percentage_weight': 60,
            'num_contours': 4
        }) # HC
        # assert listofnamedtuples(contours) # TODO: GET TO THIS LATER
        comp_data.append(contours)

    metric = evaluator(comp_data[::10], gt_data) # 10 is HC for 30 fps video, 3 fps labels
    return metric

def read_csv(filename):
    box_contours = []
    with open(filename, "r") as csv1:
        reader = csv.reader(csv1)
        next(reader)
        for row in reader:
            row = [int(float(i)) for i in row]
            row[2] -= row[4] # HC
            row[6] -= row[8] # HC
            row[10] -= row[12] # HC
            row[14] -= row[16] # HC
            # get list of every four numbers in each row
            bounding_boxes = [row[5: 9], row[13: 17], row[1: 5], row[9: 13]] # HC
            # bounding_boxes = [row[i:i + 4] for i in range(1, len(row), 4)] # last 4 is HC
            box_contours.append(DiceBoxes(*bounding_boxes, *[None for _ in range(4 - len(bounding_boxes))])) # 4 is HC
    return box_contours

def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)

def listofnamedtuples(lst):
    for k, v in list(lst.items()):
        if type(k) != str or type(v) != np.ndarray:
            return False
    return True

def box_to_coords(box):
    return ((box[0], box[1]), (box[0] + box[2], box[1]), (box[0] + box[2], box[1] + box[3]), (box[0], box[1] + box[3]))

def intersection_over_union(polyA, polyB):
    if polyA is None or polyB is None:
        return 0.0
    polygonA_shape = Polygon(box_to_coords(4 * np.array(polyA))) # HC
    polygonB_shape = Polygon(box_to_coords(polyB))
    if polygonA_shape.area == 0.0 or polygonB_shape.area == 0.0:
        return 0.0

    polygon_intersection = polygonA_shape.intersection(polygonB_shape).area
    polygon_union = polygonA_shape.area + polygonB_shape.area - polygon_intersection #inclusion exclusion

    IOU = polygon_intersection / polygon_union
    return IOU


def intersection_over_union_boxes(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def euclidean_distance(pointA, pointB):
    return sqrt(pow(pointA[0] - pointB[0], 2) + pow(pointA[1] - pointB[1], 2))

def evaluator(contour_list, ground_truth):
    num_frames = min(len(ground_truth), len(contour_list)) # HACKY Solution
    metric = 0
    if len(contour_list[0]) > 1:
        for cont_type in DiceBoxes._fields:
            metric += sum([intersection_over_union(getattr(contour_list[i], cont_type), getattr(ground_truth[i], cont_type)) for i in
                          range(num_frames)])
    elif len(contour_list[0]) == 1:
        assert len(contour_list[0]['gate_box']) == len(ground_truth[0])
        metric = sum([intersection_over_union(4 * contour_list[i]['gate_box'], ground_truth[i]) for i in range(num_frames)])
    return metric / num_frames

if __name__ == '__main__':
    test_segmentation()