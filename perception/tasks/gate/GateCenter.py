from GateSegmentationAlgo2 import GateSegmentationAlgo
from GatePerceiver import GatePerceiver
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import math
import cv2 as cv
import time
import cProfile
import statistics

class GateCenter(GatePerceiver):
    center_x_locs, center_y_locs = [], []

    def __init__(self):
        super()
        self.gate_center = self.output_class(250, 250)
        self.use_optical_flow = False
        self.optical_flow_c = 0.1
        self.gate = GateSegmentationAlgo()
    
    def analyze(self, frame, debug):
        global prvs
        rect1, rect2, debug_filter = self.gate.analyze(frame, True)
        if rect1 and rect2:
            self.gate_center = self.get_center(rect1, rect2, frame)
            if self.use_optical_flow:
                cv.circle(debug_filter, self.gate_center, 5, (3,186,252), -1)
            else:
                cv.circle(debug_filter, self.gate_center, 5, (0,0,255), -1)
        
        if debug:
            return (self.output_class(self.gate_center[0], self.gate_center[1]), debug_filter)
        return self.output_class(self.gate_center[0], self.gate_center[1])

    def center_without_optical_flow(self, center_x, center_y):
		# get starting center location, averaging over the first 2510 frames
        if len(self.center_x_locs) == 0:
            self.center_x_locs.append(center_x)
            self.center_y_locs.append(center_y)
            
        elif len(self.center_x_locs) < 25:
            self.center_x_locs.append(center_x)
            self.center_y_locs.append(center_y)
            center_x = int(statistics.mean(self.center_x_locs))
            center_y = int(statistics.mean(self.center_y_locs))
        
        # use new center location only when it is close to the previous valid location
        else:
            self.center_x_locs.append(center_x)
            self.center_y_locs.append(center_y)
            self.center_x_locs.pop(0)
            self.center_y_locs.pop(0)
            x_temp_avg = int(statistics.mean(self.center_x_locs))
            y_temp_avg = int(statistics.mean(self.center_y_locs))
            if math.sqrt((center_x - x_temp_avg)**2 + (center_y - y_temp_avg)**2) > 10:
                center_x, center_y = int(x_temp_avg), int(y_temp_avg)
                
        return (center_x, center_y)
    
    def dense_optical_flow(self, frame, prvs):
        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        mag = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        # hsv[...,0] = ang*180/np.pi
        # hsv[...,2] = mag
        # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        # cv.imshow('bgr', bgr)
        return next, mag, ang
    
    def get_center(self, rect1, rect2, frame):
        global prvs
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        center_x, center_y = (x1+x2)//2, ((y1+h1//2)+(y2+h2//2))//2
        prvs, mag, ang = self.dense_optical_flow(frame, prvs)
        # print(np.mean(mag))
        if len(self.center_x_locs) < 25 or (np.mean(mag) < 40 and ((not self.use_optical_flow ) or \
            (self.use_optical_flow and (center_x - self.gate_center[0])**2 + (center_y - self.gate_center[1])**2 < 50))):
            self.use_optical_flow = False
            return self.center_without_optical_flow(center_x, center_y)
        else:
            self.use_optical_flow = True
            return (int(self.gate_center[0] + self.optical_flow_c * np.mean(mag * np.cos(ang))), \
                (int(self.gate_center[1] + self.optical_flow_c * np.mean(mag * np.sin(ang)))))

# this part is temporary and will be covered by other files in the future
if __name__ == '__main__':
    cap = cv.VideoCapture(sys.argv[1])
    ret_tries = 0
    start_time = time.time()
    frame_count = 0
    paused = False
    speed = 1
    ret, frame1 = cap.read()
    frame1 = cv.resize(frame1, None, fx=0.3, fy=0.3)
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    gate_center = GateCenter()
    while ret_tries < 50:
        for _ in range(speed):
            ret, frame = cap.read()
        if frame_count == 1000:
            break
        if ret:
            frame = cv.resize(frame, None, fx=0.3, fy=0.3)
            center, filtered_frame = gate_center.analyze(frame, True)
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