from GatePerceiver import GatePerceiver
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(__file__))


from segmentation.combinedFilter import init_combined_filter
import numpy as np
import cv2 as cv
import time
import cProfile
import statistics

class GateSegmentationAlgo(GatePerceiver):
	center_x_locs, center_y_locs = [], []

	def __init__(self, alpha):
		super()

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
		filtered_frame = combined_filter(frame, display_figs=False)
		
		max_brightness = max([b for b in filtered_frame[:, :, 0][0]])
		lowerbound = max(0.84*max_brightness, 120)
		upperbound = 255
		_,thresh = cv.threshold(filtered_frame,lowerbound, upperbound, cv.THRESH_BINARY)
		debug_filter = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
		
		cnt = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]

		area_diff = []
		area_cnts = []

        # remove all contours with zero area
		cnt = [cnt[i] for i in range(len(cnt)) if cv.contourArea(cnt[i]) > 0]
		
		for i in range(len(cnt)):
			area_cnt = cv.contourArea(cnt[i])
			area_cnts.append(area_cnt)
			area_rect = cv.boundingRect(cnt[i])[-2] * cv.boundingRect(cnt[i])[-1]
			area_diff.append(abs((area_rect - area_cnt)/area_cnt))
			
		if len(area_diff) >= 2:
			largest_area_idx = [area_cnts.index(sorted(area_cnts, reverse=True)[i]) for i in range(min(3, len(cnt)))]
			area_diff_copy = sorted([area_diff[i] for i in largest_area_idx])
			min_i1, min_i2 = area_diff.index(area_diff_copy[0]), area_diff.index(area_diff_copy[1])
			
			(x1, y1, w1, h1) = cv.boundingRect(cnt[min_i1])
			(x2, y2, w2, h2) = cv.boundingRect(cnt[min_i2])
			cv.rectangle(debug_filter, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)
			cv.rectangle(debug_filter, (x2, y2), (x2+w2, y2+h2), (0,255,0), 2)

            # drawing center dot
			center_x, center_y = (x1+x2)//2, ((y1+h1//2)+(y2+h2//2))//2
			gate_center = self.get_actual_center(center_x, center_y)
			cv.circle(debug_filter, gate_center, 5, (0,0,255), -1)

		if debug:
			return (self.output_class(gate_center[0], gate_center[1]), debug_filter)
		return self.output_class(gate_center[0], gate_center[1])
	
	def get_actual_center(self, center_x, center_y):
		# get starting center location, averaging over the first 2510 frames
		if len(self.center_x_locs) == 0:
			self.center_x_locs.append(center_x)
			self.center_y_locs.append(center_y)
		
		elif len(self.center_x_locs) < 25:
			self.center_x_locs.append(center_x)
			self.center_y_locs.append(center_y)
			center_x = int(statistics.mean(self.center_x_locs))
			center_y = int(statistics.mean(self.center_y_locs))
        
        # use new center location only when it is close to the previous valid location
		else:
			if abs(center_x - self.center_x_locs[-1]) > 10 or \
				abs(center_y - self.center_y_locs[-1]) > 10:
				center_x, center_y = self.center_x_locs[-1], self.center_y_locs[-1]
			else:
				self.center_x_locs.append(center_x)
				self.center_y_locs.append(center_y)
		
		return (center_x, center_y)

# this part is temporary and will be covered by other files in the future
if __name__ == '__main__':
	combined_filter = init_combined_filter()
	cap = cv.VideoCapture(sys.argv[1])
	ret_tries = 0
	gate_task = GateSegmentationAlgo(0.1)
	# once = False
	start_time = time.time()
	frame_count = 0
	paused = False
	speed = 1
	while ret_tries < 50:
		ret, frame = cap.read()
		if frame_count == 1000:
			break
		if ret:
			frame = cv.resize(frame, None, fx=0.3, fy=0.3)


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
