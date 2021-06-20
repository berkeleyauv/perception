import csv

import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.dice.DiceDetector import DiceDetector
from perception.tasks.segmentation.COMB_SAL_BG import COMB_SAL_BG


class DiceCSV(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(heuristic_threshold=((5, 255), 35), run_both=((0, 1), 0),
                         centroid_distance_weight=((0, 200), 1), area_percentage_weight=((0, 200), 60),
                         num_contours=((1, 5), 1))
        self.time = 0
        self.dice_labels = open('../misc/DiceLabels.csv')
        self.dice_reader = csv.reader(self.dice_labels)
        self.row = next(self.dice_reader)
        self.dice_detector = DiceDetector()
        next(self.dice_reader)

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        _, frame = self.dice_detector.analyze(frame, True, slider_vals)
        frame = frame[0]
        if self.time % 10 == 0:
            row = next(self.dice_reader)
            print(row)
            self.row = [int(float(i)) for i in row]
            # frame = cv.putText(frame, str(row), (100, 250), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv.LINE_AA)
            # 1 2 3 4 my order
            # 3 1 4 2
            self.row[2] -= self.row[4]
            self.row[6] -= self.row[8]
            self.row[10] -= self.row[12]
            self.row[14] -= self.row[16]
        frame = cv.rectangle(frame, (self.row[1] // 4, self.row[2] // 4),
                             ((self.row[1] + self.row[3]) // 4, (self.row[2] + self.row[4]) // 4), (255, 0, 0), 2)
        frame = cv.rectangle(frame, (self.row[5] // 4, self.row[6] // 4),
                             ((self.row[5] + self.row[7]) // 4, (self.row[6] + self.row[8]) // 4), (0, 255, 0), 2)
        frame = cv.rectangle(frame, (self.row[9] // 4, self.row[10] // 4),
                             ((self.row[9] + self.row[11]) // 4, (self.row[10] + self.row[12]) // 4), (0, 0, 255), 2)
        frame = cv.rectangle(frame, (self.row[13] // 4, self.row[14] // 4),
                             ((self.row[13] + self.row[15]) // 4, (self.row[14] + self.row[16]) // 4), (0, 0, 0), 2)
        # if self.time >= 1500:
        #     self.dice_labels.close()
        self.time += 1
        return 0, [frame]