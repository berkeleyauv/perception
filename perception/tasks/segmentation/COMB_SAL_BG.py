import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.saliency_detection.MBD import MBD
from perception.vis.TestTasks.BackgroundRemoval import BackgroundRemoval
from shapely.geometry import Polygon


class COMB_SAL_BG(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(heuristic_threshold=((5,255), 35), run_both=((0,1),0), 
            centroid_distance_weight=((0,200),1), area_percentage_weight=((0,200),60), num_contours=((1,5), 1))
        self.sal = MBD()
        self.bg = BackgroundRemoval()
        self.use_saliency = True
        self.prev_centroid = (0,0)
        self.changed = True
        self.centroid_distance_weight = 1
        self.area_percentage_weight = 60
        self.num_contours = 1

    def filter_contours(self, contours, contour_filter):
        return filter(contour_filter, contours)

    def largest_contour(self, contours):
        return max(contours, key=cv.contourArea)

    def largest_contours(self, contours):
        contours.sort(key=cv.contourArea)
        num_selected = self.num_contours
        if len(contours) < num_selected:
            num_selected = len(contours)
        return contours[-self.num_contours:]

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

    def compute_heuristic(self, point1, point2, contour_area, img_area):
        return self.centroid_distance_weight * (abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])) + self.area_percentage_weight * contour_area / img_area

    def analyze_specific_img(self, frame: np.ndarray, algorithm, debug: bool, slider_vals:Dict[str, int]):
        analysis = algorithm(frame, debug, slider_vals=slider_vals)[0]
        _, threshold = cv.threshold(analysis, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        biggest_contours = self.largest_contours(contours)
        if debug:
            contour_frame = np.copy(frame)
            cv.drawContours(contour_frame, biggest_contours, -1, (255,0,0))
            return biggest_contours, [analysis, contour_frame]
        return biggest_contours, [analysis]
    
    def set_num_contours(self, num_contours):
        self.num_contours = num_contours

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        if not slider_vals['run_both'] or not debug:
            if self.use_saliency:
                returned_contour, analysis = self.analyze_specific_img(frame, self.sal.analyze, debug, slider_vals)
            else:
                returned_contour, analysis = self.analyze_specific_img(frame, self.bg.analyze, debug, slider_vals)
            largest_contour = returned_contour[0]
            M = cv.moments(largest_contour)
            if M["m00"] == 0:
                used_centroid = (int(M["m10"]), int(M["m00"]))
            else:
                used_centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
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
            returned_contour = [returned_contour]
        
        if self.changed:
            self.changed = False
        else:
            if self.compute_heuristic(used_centroid, self.prev_centroid, cv.contourArea(returned_contour[0]), 
                np.shape(frame)[0] * np.shape(frame)[1]) > slider_vals['heuristic_threshold']:
                self.switch_algorithm()
        self.prev_centroid = used_centroid

        self.centroid_distance_weight = slider_vals['centroid_distance_weight']
        self.area_percentage_weight = slider_vals['area_percentage_weight']
        self.num_contours = slider_vals['num_contours']
        if debug:
            cv.circle(frame, used_centroid, 5, (0, 255, 0), 3)
            cv.drawContours(frame, returned_contour, -1, (0,255,0))
            if slider_vals['run_both']:
                return returned_contour, [frame, sal, bg, bg_frame, sal_frame]
            else:
                return returned_contour, [frame] + analysis
        return returned_contour