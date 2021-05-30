import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.COMB_SAL_BG import COMB_SAL_BG

class DiceDetector(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(heuristic_threshold=((5,255), 35), run_both=((0,1),0), 
            centroid_distance_weight=((0,200),1), area_percentage_weight=((0,200),60), num_contours=((1,5), 1))
        self.sal = COMB_SAL_BG()
        self.sal.set_num_contours(4)
        self.sal.use_saliency = False
        
    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        contours = self.sal.analyze(frame, False, slider_vals)
        if len(contours) > 0:
            for contour in contours:
                x,y,w,h = cv.boundingRect(contour)
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)            
        return contours, [frame]