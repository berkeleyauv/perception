from TaskPerceiver import TaskPerceiver
from typing import Tuple
import sys
import os
from pathlib import Path
from collections import namedtuple
sys.path.append(str(Path(__file__).parents[2]) + '/tasks')

from segmentation.combinedFilter import init_combined_filter
import numpy as np
import cv2 as cv
import time
import cProfile

class GateSegmentationAlgo(TaskPerceiver):
	__past_centers = []
	__ema = None
	output_class = namedtuple("GateOutput", ["centerx", "centery"])
	output_type = {'centerx': np.int16, 'centery': np.int16}

	def __init__(self, alpha=0.1):
		super()
		self.__alpha = alpha
		self.combined_filter = init_combined_filter()

	def analyze(self, frame: np.ndarray, debug: bool) -> Tuple[float, float]:
		"""Takes in the background removed image and returns the center between
		the two gate posts.
		Args:
			frame: The background removed frame to analyze
			debug: Whether or not tot display intermediate images for debugging
		Reurns:
			(x,y) coordinate with center of gate
		"""
		gate_center = self.output_class(250, 250)
		filtered_frame = self.combined_filter(frame, display_figs=False)
		filtered_frame_copies = [filtered_frame for _ in range(3)]
		stacked_filter_frames = np.concatenate(filtered_frame_copies, axis = 2)
		mask = cv.inRange(stacked_filter_frames,
			np.array([100, 100, 100]), np.array([255, 255, 255]))
		_, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		if contours:
			contours.sort(key=self.findStraightness, reverse=True)
			cnts = contours[:2]
			rects = [cv.minAreaRect(c) for c in cnts]
			centers = [np.array(r[0]) for r in rects]
			boxpts = [cv.boxPoints(r) for r in rects]
			box = [np.int0(b) for b in boxpts]
			for b in box:
				cv.drawContours(stacked_filter_frames,[b],0,(0,0,255),5)
			if len(centers) >= 2:
				gate_center = (centers[0] + centers[1]) * 0.5
				if self.__ema is None:
					self.__ema = gate_center
				else:
					self.__ema = self.__alpha*gate_center + (1 - self.__alpha)*self.__ema
				gate_center = (int(self.__ema[0]), int(self.__ema[1]))
				# if len(self.__past_centers) < 15:
				# 	self.__past_centers += [gate_center]
				# else:
				# 	self.__past_centers.pop(0)
				# 	self.__past_centers += [gate_center]
				# gate_center = sum(self.__past_centers) / len(self.__past_centers)
				# gate_center = (int(gate_center[0]), int(gate_center[1]))
				cv.circle(stacked_filter_frames, gate_center, 10, (0,255,0), -1)

		if debug:
			return (self.output_class(gate_center[0], gate_center[1]), [stacked_filter_frames])
		return self.output_class(gate_center[0], gate_center[1])

	def findStraightness(self, contour): # output number = contour area/convex area, the bigger the straightest
		hull = cv.convexHull(contour, False)
		contour_area = cv.contourArea(contour)
		hull_area = cv.contourArea(hull)
		return 10 * contour_area - 5 * hull_area

# this part is temporary and will be covered by other files in the future
if __name__ == '__main__':
	combined_filter = init_combined_filter()
	cap = cv.VideoCapture(sys.argv[1])
	ret_tries = 0
	gate_task = GateSegmentationAlgo(0.1)
	# once = False
	start_time = time.time()
	frame_count = 0
	while ret_tries < 50:
		ret, frame = cap.read()
		if frame_count == 1000:
			break
		if ret:
			frame = cv.resize(frame, None, fx=0.4, fy=0.4)


			### FUNCTION CALL, can change this
			center, filtered_frame = gate_task.analyze(frame, True)
			# cProfile.run("gate_task.analyze(frame, True)")
			# cv.putText(frame, "x: %.2f" % x + " y: %.2f" % y,
			# 	(20, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX,
			# 	2.0, (0, 165, 255), 3)
			cv.imshow('original', frame)
			cv.imshow('filtered_frame', filtered_frame)
			# if not once:
			# 	print(filtered_frame)
			# 	once = True
			ret_tries = 0
			k = cv.waitKey(60) & 0xff
			if k == 27:
				break
		else:
			ret_tries += 1
		frame_count += 1
		#print(frame_count / (time.time() - start_time))
