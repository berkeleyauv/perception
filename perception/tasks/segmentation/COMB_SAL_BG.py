import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.saliency_detection.MBD import MBD
from perception.vis.TestTasks.BackgroundRemoval import BackgroundRemoval
from shapely.geometry import Polygon


class COMB_SAL_BG(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(switch_threshold=((5,255), 10))
        self.sal = MBD()
        self.bg = BackgroundRemoval()
        self.use_saliency = True
        self.prev_centroid = (0,0)
        self.changed = True

    def filter_contours(self, contours, contour_filter):
        return filter(contour_filter, contours)

    def largest_contour(self, contours):
        return max(contours, key=cv.contourArea)

    def intersection_over_union(polyA, polyB):
        polygonA_shape = Polygon(polyA)
        polygonB_shape = Polygon(polyB)

        polygon_intersection = polygonA_shape.intersection(polygonB_shape).area
        polygon_union = polygonA_shape.area + polygonB_shape.area - polygon_intersection #inclusion exclusion

        IOU = polygon_intersection / polygon_union
        return IOU

    def switch_algorithm(self):
        self.use_saliency = not self.use_saliency
        self.changed = True

    def compute_centroid_change(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        sal = self.sal.analyze(frame, debug, slider_vals=slider_vals)[0]
        bg = self.bg.analyze(frame,debug, slider_vals=slider_vals)[0]

        #BG Contours
        _, bg_threshold = cv.threshold(bg, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        bg_contours, _ = cv.findContours(bg_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bg_frame = np.copy(frame)

        #Sal Contours
        _, sal_threshold = cv.threshold(sal, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        sal_contours, _ = cv.findContours(sal_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        sal_frame = np.copy(frame)
        
        largest_sal_contour = self.largest_contour(sal_contours)
        largest_bg_contour = self.largest_contour(bg_contours)

        cv.drawContours(sal_frame, [largest_sal_contour], -1, (255,0,0))
        cv.drawContours(bg_frame, [largest_bg_contour], -1, (0,0,255))
        
        M = cv.moments(largest_sal_contour)
        centroid_sal = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv.circle(sal_frame, centroid_sal, 5, (255, 0, 0), 3)

        M_bg = cv.moments(largest_bg_contour)
        centroid_bg = (int(M_bg["m10"] / M_bg["m00"]), int(M_bg["m01"] / M_bg["m00"]))
        cv.circle(bg_frame, centroid_bg, 5, (0, 0, 255), 3)

        
        if self.use_saliency:
            returned_contour = largest_sal_contour
            used_centroid = centroid_sal
        else:
            returned_contour = largest_bg_contour
            used_centroid = centroid_bg
        
        if self.changed:
            self.changed = False
        else:
            if self.compute_centroid_change(used_centroid, self.prev_centroid) > slider_vals['switch_threshold']:
                self.switch_algorithm()
                print('Switched!')
        self.prev_centroid = used_centroid
        cv.circle(frame, used_centroid, 5, (0, 255, 0), 3)
        cv.drawContours(frame, [returned_contour], -1, (0,255,0))
        return returned_contour, [frame, sal, bg, bg_frame, sal_frame]
