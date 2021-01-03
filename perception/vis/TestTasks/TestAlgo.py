from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class TestAlgo(TaskPerceiver):
	def __init__(self):
		super().__init__(canny_low=((0, 255), 100), canny_high=((0, 255), 200))

	def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
		fig = plt.figure()
		x1 = np.linspace(0.0, 5.0)
		x2 = np.linspace(0.0, 2.0)

		y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
		y2 = np.cos(2 * np.pi * x2)

		line1, = plt.plot(x1, y1, 'ko-')
		line1.set_ydata(np.cos(2 * np.pi * (x1 + slider_vals['canny_low'] * 3.14 / 2)) * np.exp(-x1))
		fig.canvas.draw()

		return frame, [frame, cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.ROTATE_180),
					   cv.Canny(frame, slider_vals['canny_low'], slider_vals['canny_high']),
					   cv.flip(cv.Canny(frame, slider_vals['canny_low'], slider_vals['canny_high']), 0), fig]