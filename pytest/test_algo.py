import csv
import numpy as np
from math import sqrt
from shapely.geometry import Polygon
from perception.vis.FrameWrapper import FrameWrapper
from perception.tasks.gate.GateSegmentationAlgoA import GateSegmentationAlgoA
import os

class TestAlgo:
    def test_segmentation(self):
        print(os.path.dirname(os.path.abspath(__file__)))
        data_sources = ["../perception/vis/datasets/GOPR1142.mp4"]
        gt_filename = "labels/GOPR1142-gate_tracked_all.csv"
        best_metric = 0.01
        assert self.run_algo(data_sources, GateSegmentationAlgoA(), gt_filename) >= best_metric

    def isnamedtupleinstance(self, x):
        t = type(x)
        b = t.__bases__
        if len(b) != 1 or b[0] != tuple: return False
        f = getattr(t, '_fields', None)
        if not isinstance(f, tuple): return False
        return all(type(n)==str for n in f)

    def listofnamedtuples(self, lst):
        for k, v in list(lst.items()):
            if type(k) != str or type(v) != tuple:
                return False
        return True

    def read_csv(self, filename):
        box_contours = []
        with open(filename, "r") as csv1:
            reader = csv.reader(csv1, delimiter=" ")
            for row in reader:
                    coords = np.array(row)[3:7]
                    contours = [[coords[0], coords[1]], [coords[0] + coords[3], coords[1]], [coords[0], coords[1] + coords[2]], [coords[0] + coords[3], coords[1] + coords[2]]]
                    box_contours.append(contours)
        return box_contours


    def intersection_over_union(self, polyA, polyB):
        polygonA_shape = Polygon(polyA)
        polygonB_shape = Polygon(polyB)

        polygon_intersection = polygonA_shape.intersection(polygonB_shape).area
        polygon_union = polygonA_shape.area + polygonB_shape.area - polygon_intersection #inclusion exclusion

        IOU = polygon_intersection / polygon_union
        return IOU


    def intersection_over_union_boxes(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = (xB - xA) * (yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def euclidean_distance(self, pointA, pointB):
        return sqrt(pow(pointA[0] - pointB[0], 2) + pow(pointA[1] - pointB[1], 2))

    def evaluator(self, contour_list, ground_truth):
        num_frames = len(contour_list)
        metric = 0
        if len(contour_list[0]) == 2:
            comparator = []
            for frame in ground_truth:
                xCenter, yCenter = 0.5 * (ground_truth[0][0] + ground_truth[1][0]), 0.5 * (ground_truth[0][1] + ground_truth[2][1])
                comparator.append([xCenter, yCenter])
            assert len(contour_list[0]) == len(comparator[0])
            metric = sum([self.euclidean_distance(contour_list[i]['gate_box'], comparator[i]) for i in range(num_frames)])
        elif len(contour_list[0]) == 4:
            assert len(contour_list[0]) == len(ground_truth[0])
            metric = sum([self.intersection_over_union(contour_list[i], ground_truth[i]) for i in range(num_frames)])
        return metric / num_frames

    def run_algo(self, data_source, algorithm, gt_filename):
        data = FrameWrapper(data_source, 0.25)
        comp_data = []

        for frame in data:
            contours = algorithm.analyze(frame)
            assert self.listofnamedtuples(contours)
            comp_data.append(contours)

        gt_data = self.read_csv(gt_filename)

        metric = self.evaluator(comp_data, gt_data)
        print('FINAL METRIC: ', metric)
        return metric

if __name__ == '__main__':
    a = TestAlgo()
    # a.test_segmentation()
    a.test_segmentation()