import numpy as np
import cv2 as cv
import math
from typing import Dict, Tuple

# Get input from webcam
#cap = cv.VideoCapture(0)
def nothing(x):
	pass
#cap = cv.VideoCapture(0)
"""
cv.namedWindow('contours')
cv.createTrackbar('blow','contours',0,255,nothing)
cv.createTrackbar('glow','contours',0,255,nothing)
cv.createTrackbar('rlow','contours',0,255,nothing)
cv.createTrackbar('bhigh','contours',0,255,nothing)
cv.createTrackbar('ghigh','contours',0,255,nothing)
cv.createTrackbar('rhigh','contours',0,255,nothing)
cv.setTrackbarPos('bhigh','contours',255)
cv.setTrackbarPos('ghigh','contours',255)
cv.setTrackbarPos('rhigh','contours',255)
"""
class Visualizer:
	def __init__(self, vars: Dict[str, Tuple[Tuple[int, int], int]]):
		self.variables = vars.keys()
		cv.namedWindow('Debug Frames')
		for name, info in vars.items():
			range, default_val = info
			low_range, high_range = range
			cv.createTrackbar(name, 'Debug Frames', low_range, high_range, nothing)
			cv.setTrackbarPos(name, 'Debug Frames', default_val)

	def display(self, frames):
		num_frames = len(frames)
		assert (num_frames > 0 and num_frames <= 9), 'Invalid number of frames!'

		columns = math.ceil(num_frames/math.sqrt(num_frames))
		rows = math.ceil(num_frames/columns)
		print('cols', columns, 'rows', rows, 'num_frames', num_frames)
		frame_num = 0
		to_show = 0
		for j in range(rows):
			this_row = frames[frame_num]
			for i in range(1, columns):
				frame_num += 1
				print('frame_num', frame_num, j, i)
				if frame_num < num_frames:
					#print(this_row.shape, frames[frame_num].shape)
					this_row = np.hstack((this_row, frames[frame_num]))
				else:
					print("here!")
					this_row = np.hstack((this_row, np.zeros(frames[0].shape)))
			if type(to_show) != int:
				to_show = np.vstack((to_show, this_row))
			else:
				to_show = this_row
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
