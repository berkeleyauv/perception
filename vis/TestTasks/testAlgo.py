from TaskPerceiver import TaskPerceiver
from typing import Tuple
import sys
import os
import numpy as np
import cv2 as cv

class TestAlgo(TaskPerceiver):
	def analyze(self, frame: np.ndarray, debug: bool) -> Tuple[np.ndarray, np.ndarray]:
		return frame, [cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.Canny(frame, 100, 200)]
