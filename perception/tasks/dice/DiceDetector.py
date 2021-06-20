import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.COMB_SAL_BG import COMB_SAL_BG
from perception.tasks.ReturnStructs import DiceBoxes
import queue

class DiceDetector(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(heuristic_threshold=((5,255), 35), run_both=((0,1),0), 
            centroid_distance_weight=((0,200),1), area_percentage_weight=((0,200),60), num_contours=((1,5), 1))
        self.sal = COMB_SAL_BG()
        self.sal.set_num_contours(1)
        self.sal.use_saliency = False
        self.interpolator = RectangleInterpolator(20)
        
    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        contours = self.sal.analyze(frame, False, slider_vals)
        boxes = []
        if len(contours) > 0:
            for contour in contours:
                x,y,w,h = cv.boundingRect(contour)
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                boxes.append([x, y, w, h])
                self.interpolator.insert_new_rectangle(x, y, w, h)
        diceBoxes = DiceBoxes(*boxes, *[None for _ in range(4 - len(boxes))])
        avg_coord1, avg_coord2 = self.interpolator.get_avg_coordinates()
        cv.rectangle(frame, avg_coord1, avg_coord2, (255, 0, 255), 2)
        if debug:
            return diceBoxes, [frame]
        return diceBoxes

class RectangleInterpolator():

    def __init__(self, num_rectangles):
        self.contour_queue = queue.Queue(maxsize=num_rectangles)
        self.coord1Sum = (0, 0)
        self.coord2Sum = (0, 0)

    def insert_new_rectangle(self, x, y, w, h):
        if self.contour_queue.full():
            value = self.contour_queue.get()
            self.coord1Sum = (self.coord1Sum[0] - value[0][0], self.coord1Sum[1] - value[0][1])
            self.coord2Sum = (self.coord2Sum[0] - value[1][0], self.coord2Sum[1] - value[1][1])
        self.contour_queue.put([(x, y), (x+w, y+h)])
        self.coord1Sum = (self.coord1Sum[0] + x, self.coord1Sum[1] + y)
        self.coord2Sum = (self.coord2Sum[0] + x + w, self.coord2Sum[1] + y + h)
    
    def get_avg_coordinates(self):
        avg_coord1 = (int(self.coord1Sum[0] / self.contour_queue.qsize()), int(self.coord1Sum[1] / self.contour_queue.qsize()))
        avg_coord2 = (int(self.coord2Sum[0] / self.contour_queue.qsize()), int(self.coord2Sum[1] / self.contour_queue.qsize()))
        return avg_coord1, avg_coord2
