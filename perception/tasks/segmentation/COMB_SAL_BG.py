import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.saliency_detection.MBD import MBD
from perception.vis.TestTasks.BackgroundRemoval import BackgroundRemoval

class COMB_SAL_BG(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(blur = ((0, 10), 2), lamda = ((0,10),1), lower_bound=((0,255), 10))
        self.sal = MBD()
        self.bg = BackgroundRemoval()
    
    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        sal = self.sal.analyze(frame, debug, slider_vals=slider_vals)
        bg = self.bg.analyze(frame,debug, slider_vals=slider_vals)
        # print(type(sal), type(bg))
        ret = np.zeros(sal[0].shape)
        cv.bitwise_and(sal[0],bg[0], dst=ret)
        # small = min(cv.s)
        # print (type(ret))
        return frame, [frame, sal[0], bg[0], ret]