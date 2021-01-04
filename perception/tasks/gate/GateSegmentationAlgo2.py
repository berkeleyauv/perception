from TaskPerceiver import TaskPerceiver
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(__file__))


from ..segmentation.combinedFilter import init_combined_filter
import numpy as np
import math
import cv2 as cv
import time
import cProfile
import statistics

class GateSegmentationAlgo(TaskPerceiver):
    center_x_locs, center_y_locs = [], []
    
    def __init__(self):
        super()
        self.combined_filter = init_combined_filter()

    def analyze(self, frame: np.ndarray, debug: bool) -> Tuple[float, float]:
        """Takes in the background removed image and returns the center between
        the two gate posts.
        Args:
            frame: The background removed frame to analyze
            debug: Whether or not tot display intermediate images for debugging
		Reurns:
			(x,y) coordinate with center of gate
		"""
        rect1, rect2 = None, None

        filtered_frame = self.combined_filter(frame, display_figs=False)
		
        max_brightness = max([b for b in filtered_frame[:, :, 0][0]])
        lowerbound = max(0.84*max_brightness, 120)
        upperbound = 255
        _,thresh = cv.threshold(filtered_frame,lowerbound, upperbound, cv.THRESH_BINARY)
        debug_filter = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        
        cnt = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
        
        area_diff = []
        area_cnts = []

        # remove all contours with zero area
        cnt = [cnt[i] for i in range(len(cnt)) if cv.contourArea(cnt[i]) > 0]
		
        for i in range(len(cnt)):
            area_cnt = cv.contourArea(cnt[i])
            area_cnts.append(area_cnt)
            area_rect = cv.boundingRect(cnt[i])[-2] * cv.boundingRect(cnt[i])[-1]
            area_diff.append(abs((area_rect - area_cnt)/area_cnt))
			
        if len(area_diff) >= 2:
            largest_area_idx = [area_cnts.index(sorted(area_cnts, reverse=True)[i]) for i in range(min(3, len(cnt)))]
            area_diff_copy = sorted([area_diff[i] for i in largest_area_idx])
            min_i1, min_i2 = area_diff.index(area_diff_copy[0]), area_diff.index(area_diff_copy[1])
			
            rect1 = cv.boundingRect(cnt[min_i1])
            rect2 = cv.boundingRect(cnt[min_i2])
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            cv.rectangle(debug_filter, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)
            cv.rectangle(debug_filter, (x2, y2), (x2+w2, y2+h2), (0,255,0), 2)
        
        if debug:
            return (rect1, rect2, debug_filter)
        return (rect1, rect2)
        
    

# this part is temporary and will be covered by other files in the future
if __name__ == '__main__':
    cap = cv.VideoCapture(sys.argv[1])
    ret_tries = 0
    start_time = time.time()
    frame_count = 0
    paused = False
    speed = 1
    gate_task = GateSegmentationAlgo()
    while ret_tries < 50:
        for _ in range(speed):
            ret, frame = cap.read()
        if frame_count == 1000:
            break
        if ret:
            frame = cv.resize(frame, None, fx=0.3, fy=0.3)
            rect1, rect2, filtered_frame = gate_task.analyze(frame, True)
            cv.imshow('original', frame)
            cv.imshow('filtered_frame', filtered_frame)
            ret_tries = 0
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('i') and speed > 1:
                speed -= 1
            if key == ord('o'):
                speed += 1
        else:
            ret_tries += 1
        frame_count += 1
