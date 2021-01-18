from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class TestAlgo(TaskPerceiver):
	def __init__(self):
		super().__init__(canny_low=((0, 255), 100), canny_high=((0, 255), 200))
		self.t = .1

	def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
		fig = plt.figure()
		x = np.linspace(0.0, 5.0)
		y = np.cos(2 * np.pi * (x + slider_vals['canny_low'] * 3.14 / 2)) * np.exp(-x * self.t)
		plt.plot(x, y, 'ko-')
		fig.canvas.draw()

		self.t *= 1.01
		return frame, [frame,
					   cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
					   cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.ROTATE_180),
					   cv.Canny(frame, slider_vals['canny_low'], slider_vals['canny_high']),
					   cv.flip(cv.Canny(frame, slider_vals['canny_low'], slider_vals['canny_high']), 0),
					   fig]

if __name__ == '__main__':
	from perception.vis.vis import run
	run(['webcam'], TestAlgo(), True)