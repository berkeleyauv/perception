import numpy as np
import argparse
import cv2 as cv
import heapq
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.COMB_SAL_BG import COMB_SAL_BG

class RouletteColorDetector(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(heuristic_threshold=((5,255), 35), run_both=((0,1),0), 
            centroid_distance_weight=((0,200),1), area_percentage_weight=((0,200),60), num_contours=((1,5), 1))
        self.sal = COMB_SAL_BG()
        self.sal.set_num_contours(3)
        self.sal.use_saliency = False

    def draw_centers_of_largest_contours(self, img, n, contours, color, size, offset=(0,0), f=cv.contourArea):
        largest_n_contours = heapq.nlargest(n,contours,key=f)
        cv.drawContours(img,largest_n_contours,-1,color,size, offset=offset)
        for c in largest_n_contours:
            M = cv.moments(c)
            if M["m00"] != 0.0:
            # Offset due to the processing on an extracted image (x,y) + (center in other imagex, center in other image y)
                cX = offset[0] + int(M["m10"] / M["m00"])
                cY = offset[1] + int(M["m01"] / M["m00"])
                cv.circle(img, (cX, cY), size, color, -1)

    def heuristic(self, contour):
        rect = cv.minAreaRect(contour)
        area = rect[1][0] * rect[1][1]
        diff = cv.contourArea(cv.convexHull(contour)) - cv.contourArea(contour)
        cent = rect[0]
        heur = 3 * area - 9 * diff
        return heur

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        contours = self.sal.analyze(frame, False, slider_vals)
        if len(contours) > 0:
            single_contour_list = np.concatenate(contours)
            x,y,w,h = cv.boundingRect(single_contour_list)
            spinner = frame[y:y+h, x:x+w]

            hsv_spinner = cv.cvtColor(spinner,cv.COLOR_BGR2HSV)
            blurred_spinner = cv.bilateralFilter(hsv_spinner,7,50,50)
            bgr_blurred_spinner = cv.cvtColor(blurred_spinner,cv.COLOR_HSV2BGR)

            # Index 0: Hue, Index 1: Saturation, Index 2: Value
            individual_channels = cv.split(blurred_spinner)

            individual_channels[0] = cv.equalizeHist(individual_channels[0])
            individual_channels[1] = cv.equalizeHist(individual_channels[1])
            individual_channels[2] = cv.equalizeHist(individual_channels[2])

            equalized_image = cv.merge((individual_channels[0],individual_channels[1],individual_channels[2]))

            # Background Green Detection
            Lab_spinner = cv.cvtColor(bgr_blurred_spinner, cv.COLOR_BGR2LAB)
            Lab_individual_channels = cv.split(Lab_spinner)
            _ , L_threshold = cv.threshold(Lab_individual_channels[0],1,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
            _ , a_threshold = cv.threshold(Lab_individual_channels[1],1,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
            b_threshold_value, b_threshold_background = cv.threshold(Lab_individual_channels[2],1,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
            b_bias = 3
            b_threshold_background_value = b_bias + b_threshold_value
            _, b_threshold_background = cv.threshold(Lab_individual_channels[2],b_threshold_background_value,255,cv.THRESH_BINARY)

            b_edges_bias = 5
            b_threshold_edges = b_threshold_value - b_edges_bias
            _, b_threshold_edges = cv.threshold(Lab_individual_channels[2],b_threshold_edges,255,cv.THRESH_BINARY_INV)

            # General Thresholds
            _, hue_threshold_norm = cv.threshold(individual_channels[0],0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
            _, saturation_threshold_inv = cv.threshold(individual_channels[1],0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
            _, value_threshold_inv = cv.threshold(individual_channels[2],0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

            # red_threshold = hue_threshold_norm & saturation_threshold_inv & value_threshold_inv & !b_threshold_background
            red_threshold = cv.bitwise_and(cv.bitwise_and(hue_threshold_norm, cv.bitwise_and(saturation_threshold_inv, value_threshold_inv)),cv.bitwise_not(b_threshold_background))
            red_contours, _ = cv.findContours(red_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            self.draw_centers_of_largest_contours(frame,2,red_contours,(0,0,255),3,offset=(x,y))

            # black_threshold = hue_threshold_norm & !saturation_threshold_inv & value_threshold_inv & !b_threshold_background
            black_threshold = cv.bitwise_and(cv.bitwise_and(hue_threshold_norm, cv.bitwise_and(cv.bitwise_not(saturation_threshold_inv), value_threshold_inv)),cv.bitwise_not(b_threshold_background))
            black_contours, _ = cv.findContours(black_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            self.draw_centers_of_largest_contours(frame,2,black_contours,(0,0,0),3,offset=(x,y))

            # green_threshold = !saturation_threshold_inv & !value_threshold_inv & !b_threshold_background & !b_threshold_edges
            green_threshold = cv.bitwise_and(cv.bitwise_and(cv.bitwise_and(cv.bitwise_not(saturation_threshold_inv), cv.bitwise_not(value_threshold_inv)),cv.bitwise_not(b_threshold_background)),cv.bitwise_not(b_threshold_edges))
            kernel = np.ones((6,6),np.uint8)
            green_threshold = cv.morphologyEx(green_threshold,cv.MORPH_OPEN,kernel)
            green_contours, _ = cv.findContours(green_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            self.draw_centers_of_largest_contours(frame,2,green_contours,(0,255,0),3,offset=(x,y), f=self.heuristic)

            # Showing Spinner Bounding Box
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        return [red_contours, green_contours, black_contours], [frame]