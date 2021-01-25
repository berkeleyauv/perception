from perception.tasks.gate.GateSegmentationAlgoA import GateSegmentationAlgoA
from perception.tasks.TaskPerceiver import TaskPerceiver
from collections import namedtuple

import numpy as np
import math
import cv2 as cv
import statistics


class GateCenterAlgo(TaskPerceiver):
    center_x_locs, center_y_locs = [], []
    output_class = namedtuple("GateOutput", ["centerx", "centery"])
    output_type = {'centerx': np.int16, 'centery': np.int16}

    def __init__(self):
        super().__init__(optical_flow_c=((0, 100), 10))
        self.gate_center = self.output_class(250, 250)
        self.use_optical_flow = False
        self.optical_flow_c = 0.1
        self.gate = GateSegmentationAlgoA()
        self.prvs = None

    # TODO: do input and return typing
    def analyze(self, frame, debug, slider_vals):
        self.optical_flow_c = slider_vals['optical_flow_c']/100
        rect, debug_filters = self.gate.analyze(frame, True)
        debug_filter = debug_filters[-1]
        debug_filters = debug_filters[:-1]

        if self.prvs is None:
            # frame = cv.resize(frame, None, fx=0.3, fy=0.3)
            self.prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else: 
            if rect[0] and rect[1]:
                self.gate_center = self.get_center(rect[0], rect[1], frame)
                if self.use_optical_flow:
                    cv.circle(debug_filter, self.gate_center, 5, (3,186,252), -1)
                else:
                    cv.circle(debug_filter, self.gate_center, 5, (0,0,255), -1)
        
        if debug:
            return (self.gate_center[0], self.gate_center[1]), list(debug_filters) + [debug_filter]
        return (self.gate_center[0], self.gate_center[1])

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
    
    def dense_optical_flow(self, frame):
        next_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(self.prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # hsv[...,0] = ang*180/np.pi
        # hsv[...,2] = mag
        # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        # cv.imshow('bgr', bgr)
        return next_frame, mag, ang

    def get_center(self, rect1, rect2, frame):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        center_x, center_y = (x1 + x2) // 2, ((y1 + h1 // 2) + (y2 + h2 // 2)) // 2
        self.prvs, mag, ang = self.dense_optical_flow(frame)

        if len(self.center_x_locs) < 25 or (np.mean(mag) < 40 and ((not self.use_optical_flow ) or \
            (self.use_optical_flow and (center_x - self.gate_center[0])**2 + (center_y - self.gate_center[1])**2 < 50))):
            self.use_optical_flow = False
            return self.center_without_optical_flow(center_x, center_y)
        self.use_optical_flow = True
        return (int(self.gate_center[0] + self.optical_flow_c * np.mean(mag * np.cos(ang))), \
                (int(self.gate_center[1] + self.optical_flow_c * np.mean(mag * np.sin(ang)))))


if __name__ == '__main__':
    from perception.vis.vis import run
    run(['../../../../GOPR1142.MP4'], GateCenterAlgo(), False)