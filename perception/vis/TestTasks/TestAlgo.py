from TaskPerceiver import TaskPerceiver
from typing import Dict
import sys
import os
import numpy as np
import cv2 as cv

class TestAlgo(TaskPerceiver):
	def __init__(self):
		super().__init__(canny_low=((0, 255), 100), canny_high=((0, 255), 200))

	def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):

		return frame, [frame, cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.ROTATE_180),
					   cv.Canny(frame, slider_vals['canny_low'], slider_vals['canny_high']),
					   cv.flip(cv.Canny(frame, slider_vals['canny_low'], slider_vals['canny_high']), 0)]