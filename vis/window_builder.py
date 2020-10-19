import numpy as np
import cv2 as cv
import math
from typing import Dict, Tuple

# Get input from webcam
#cap = cv.VideoCapture(0)
def nothing(x):
	pass
#cap = cv.VideoCapture(0)

class Visualizer:
	def __init__(self, vars: Dict[str, Tuple[Tuple[int, int], int]]):
		self.variables = vars.keys()
		cv.namedWindow('Debug Frames')
		for name, info in vars.items():
			range, default_val = info
			low_range, high_range = range
			cv.createTrackbar(name, 'Debug Frames', low_range, high_range, nothing)
			cv.setTrackbarPos(name, 'Debug Frames', default_val)

	def display(self, frames, triple=False):
		num_frames = len(frames)
		assert (num_frames > 0 and num_frames <= 9), 'Invalid number of frames!'

		columns = math.ceil(num_frames/math.sqrt(num_frames))
		rows = math.ceil(num_frames/columns)
		frame_num = 0
		to_show = 0
		for j in range(rows):
			this_row = frames[frame_num]
			for i in range(columns * j + 1, columns * (j + 1)):
				frame_num += 1
				if frame_num < num_frames:
					to_add = frames[frame_num]
					this_row = np.hstack((this_row, to_add))
				else:
					this_row = np.hstack((this_row, np.zeros(frames[0].shape, dtype=np.uint8)))
			if type(to_show) != int:
				to_show = np.vstack((to_show, this_row))
			else:
				to_show = this_row
			frame_num += 1
		cv.imshow('Debug Frames', to_show)

	def update_vars(self) -> Dict[str, int]:
		variable_values = {}
		for var in self.variables:
			variable_values[var] = cv.getTrackbarPos(var, 'Debug Frames')
		return variable_values

# Continue until user ends program
if __name__ == '__main__':
	while (True):
	    ret, frame = cap.read()

	    frame = cv.resize(frame, (int(frame.shape[1]*1/2), int(frame.shape[0]*1/2)), interpolation = cv.INTER_AREA) # Downsize image
	    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

	    hs = cv.getTrackbarPos('blow','contours')
	    ss = cv.getTrackbarPos('glow','contours')
	    vs = cv.getTrackbarPos('rlow','contours')
	    hl = cv.getTrackbarPos('bhigh','contours')
	    sl = cv.getTrackbarPos('ghigh','contours')
	    vl = cv.getTrackbarPos('rhigh','contours')

	    mask = cv.inRange(hsv, np.array([hs,ss,vs]), np.array([hl,sl,vl]))
	    res = cv.bitwise_and(frame,frame, mask= mask)
	    #cv.imshow('Viz', np.vstack((np.hstack((res, res, res)), np.hstack((res, res, res)), np.hstack((res, res, res)))))




	    #cv.imshow('nine', np.vstack((np.hstack((res, res, res)), np.hstack((res, res, res)), np.hstack((res, res, res)))))



	    if cv.waitKey(1) and 0xFF == ord('q'): # Exit
	        break



	#Cleanup
	cap.release()
	cv.destroyAllWindows()
