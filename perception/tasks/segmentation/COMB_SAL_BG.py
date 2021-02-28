import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.saliency_detection.MBD import MBD
from perception.vis.TestTasks.BackgroundRemoval import BackgroundRemoval
from shapely.geometry import Polygon


class COMB_SAL_BG(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(blur = ((0, 10), 2), lamda = ((0,10),1), lower_bound=((0,255), 10))
        self.sal = MBD()
        self.bg = BackgroundRemoval()

    # def filter_contours(self, contour_main, contour_filter):
    #     TODO: find algorithm to filter contour_main based upon contour_filter

    def intersection_over_union(polyA, polyB):
        polygonA_shape = Polygon(polyA)
        polygonB_shape = Polygon(polyB)

        polygon_intersection = polygonA_shape.intersection(polygonB_shape).area
        polygon_union = polygonA_shape.area + polygonB_shape.area - polygon_intersection #inclusion exclusion

        IOU = polygon_intersection / polygon_union
        return IOU

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        sal = self.sal.analyze(frame, debug, slider_vals=slider_vals)[0]
        bg = self.bg.analyze(frame,debug, slider_vals=slider_vals)[0]
        ret = cv.bitwise_and(sal,bg)

        #Combined Contours
        _, threshold = cv.threshold(ret, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #BG Contours
        _, bg_threshold = cv.threshold(bg, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, bg_contours, _ = cv.findContours(bg_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bg_frame = np.copy(frame)

        #Sal Contours
        _, sal_threshold = cv.threshold(sal, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, sal_contours, _ = cv.findContours(sal_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        sal_frame = np.copy(frame)

        cv.drawContours(sal_frame, sal_contours, -1, (255,0,0))
        cv.drawContours(frame, contours, -1, (0,255,0))
        cv.drawContours(bg_frame, bg_contours, -1, (0,0,255))
        return frame, [frame, sal, bg, ret, bg_frame, sal_frame]
