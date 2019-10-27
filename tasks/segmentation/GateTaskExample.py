from TaskPerceiver import TaskPerceiver
from typing import Tuple
from sys import argv as args
from combinedFilter import init_combined_filter
import numpy as np
import cv2 as cv
#from segmentation.aggregateRescaling import init_aggregate_rescaling

class GateTask(TaskPerceiver):
	def analyze(self, frame: np.ndarray, debug: bool) -> Tuple[float, float]:
		"""Takes in the background removed image and returns the center between
		the two gate posts.
		Args:
			frame: The background removed frame to analyze
			debug: Whether or not tot display intermediate images for debugging
		Reurns:
			(x,y) coordinate with center of gate
		"""
		filtered_frame = combined_filter(frame, display_figs=False)
		filtered_frame_copies = [filtered_frame for _ in range(3)]
		stacked_filter_frames = np.concatenate(filtered_frame_copies, axis = 2)
		mask = cv.inRange(stacked_filter_frames,
			np.array([100, 100, 100]), np.array([255, 255, 255]))
		_, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		num_of_cnt = 2
		centers = np.array([[]])
		while contours and num_of_cnt:
			cnt = max(contours, key=self.findStraightness)#lambda x: cv.minAreaRect(x)[1][0] * cv.minAreaRect(x)[1][1])
			rect = cv.minAreaRect(cnt)
			if centers.size == 0:
				centers = np.array([rect[0]])
			else:
				centers = np.append(centers, np.array([rect[0]]), axis=0)
			boxpts = cv.boxPoints(rect)
			box = np.int0(boxpts)
			cv.drawContours(stacked_filter_frames,[box],0,(0,0,255),5)
			for corner in boxpts:
				cv.circle(stacked_filter_frames, (corner[0], corner[1]), 10, (0,0,255), -1)
			find_index = [True if e is cnt else False for e in contours]
			index = find_index.index(True)
			contours.pop(index)
			num_of_cnt -= 1
		if centers.size > 2:
			gate_center = (centers[0] + centers[1]) * 0.5
			gate_center = (int(gate_center[0]), int(gate_center[1]))
			cv.circle(stacked_filter_frames, gate_center, 10, (0,255,0), -1)
		if debug:
			return ((250, 250), stacked_filter_frames)
		return (250, 250)

	def findStraightness(self, contour): # output number = contour area/convex area, the bigger the straightest
		hull = cv.convexHull(contour, False)
		contour_area = cv.contourArea(contour)
		hull_area = cv.contourArea(hull)
		return 10 * contour_area + 5 * (hull_area - contour_area)

# this part is temporary and will be covered by other files in the future
if __name__ == '__main__':
	combined_filter = init_combined_filter()
	cap = cv.VideoCapture(args[1])
	ret_tries = True
	gate_task = GateTask()
	# once = False
	while 1 and ret_tries < 50:
		ret, frame = cap.read()
		if ret:
			frame = cv.resize(frame, None, fx=0.4, fy=0.4)


			### FUNCTION CALL, can change this
			(x, y), filtered_frame = gate_task.analyze(frame, True)
			cv.putText(frame, "x: %.2f" % x + " y: %.2f" % y,
				(20, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX,
				2.0, (0, 165, 255), 3)
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
