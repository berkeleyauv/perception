import numpy as np
import cv2
import math

# Get input from webcam
#cap = cv2.VideoCapture(0)
def nothing(x):
	pass
#cap = cv2.VideoCapture(0)
cv2.namedWindow('contours')
cv2.createTrackbar('blow','contours',0,255,nothing)
cv2.createTrackbar('glow','contours',0,255,nothing)
cv2.createTrackbar('rlow','contours',0,255,nothing)
cv2.createTrackbar('bhigh','contours',0,255,nothing)
cv2.createTrackbar('ghigh','contours',0,255,nothing)
cv2.createTrackbar('rhigh','contours',0,255,nothing)
cv2.setTrackbarPos('bhigh','contours',255)
cv2.setTrackbarPos('ghigh','contours',255)
cv2.setTrackbarPos('rhigh','contours',255)

def display(frames):
	num_frames = len(frames)
	assert (num_frames > 0 and num_frames <= 9), 'Invalid number of frames!'

	columns = math.ceil(num_frames/math.sqrt(num_frames))
	rows = math.ceil(num_frames/columns)

	frame_num = 0
	to_show = None
	for _ in range(rows):
		this_row = frames[frame_num]
		for _ in range(columns - 1):
			frame_num += 1
			if frame_num < num_frames:
				#print(this_row.shape, frames[frame_num].shape)
				this_row = np.hstack((this_row, frames[frame_num]))
			else:
				this_row = np.hstack((this_row, frames[0]))

		if to_show:
			to_show = np.vstack((to_show, this_row))
		else:
			to_show = this_row
	cv2.imshow('Viz', to_show)



# Continue until user ends program
"""
while (True):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (int(frame.shape[1]*1/2), int(frame.shape[0]*1/2)), interpolation = cv2.INTER_AREA) # Downsize image
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    hs = cv2.getTrackbarPos('blow','contours')
    ss = cv2.getTrackbarPos('glow','contours')
    vs = cv2.getTrackbarPos('rlow','contours')
    hl = cv2.getTrackbarPos('bhigh','contours')
    sl = cv2.getTrackbarPos('ghigh','contours')
    vl = cv2.getTrackbarPos('rhigh','contours')

    mask = cv2.inRange(hsv, np.array([hs,ss,vs]), np.array([hl,sl,vl]))
    res = cv2.bitwise_and(frame,frame, mask= mask)
    #cv2.imshow('Viz', np.vstack((np.hstack((res, res, res)), np.hstack((res, res, res)), np.hstack((res, res, res)))))




    #cv2.imshow('nine', np.vstack((np.hstack((res, res, res)), np.hstack((res, res, res)), np.hstack((res, res, res)))))



    if cv2.waitKey(1) and 0xFF == ord('q'): # Exit
        break



#Cleanup
cap.release()
cv2.destroyAllWindows()
"""
