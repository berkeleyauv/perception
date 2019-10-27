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

		Returns:
			(x,y) coordinate with center of gate
		"""
		filtered_frame_copies = [filtered_frame for _ in range[10]]
		np.stack(filtered_frame_copies, axis = -1)
		mask = cv.inRange(filtered_frame, np.array[190], )

		filtered_frame = combined_filter(frame, display_figs=False)
		if debug:
			return ((250, 250), filtered_frame)
		return (250, 250)

# this part is temporary and will be covered by other files in the future
if __name__ == '__main__':
	combined_filter = init_combined_filter()
	cap = cv.VideoCapture(args[1])
	ret_tries = True
	gate_task = GateTask()
	while 1 and ret_tries < 50:
		ret, frame = cap.read()
		if ret:
			frame = cv.resize(frame, None, fx=0.4, fy=0.4)
			filtered_frame = combined_filter(frame, display_figs=False)

			### FUNCTION CALL, can change this
			(x, y), filtered_frame = gate_task.analyze(frame, True)
			cv.putText(frame, "x: %.2f" % x + " y: %.2f" % y,
				(20, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX,
				2.0, (0, 165, 255), 3)
			cv.imshow('original', frame)
			cv.imshow('filtered_frame', filtered_frame)

			ret_tries = 0
			k = cv.waitKey(60) & 0xff
			if k == 27:
				break
		else:
			ret_tries += 1
