import cv2 as cv
from sys import argv as args
import numpy as np
import numpy.linalg as LA

#Jenny -> unsigned ints fixed the problem
#Damas -> flip weight vector every frame

max_value = 100

low_B = 0
low_G = 0
low_R = 0
high_B = max_value
high_G = max_value
high_R = max_value

low_B_name = 'Low B'
low_G_name = 'Low G'
low_R_name = 'Low R'
high_B_name = 'High H'
high_G_name = 'High S'
high_R_name = 'High V'

window_detection_name = 'Object Detection'

def on_low_B_thresh_trackbar(val):
	global low_B
	global high_B
	low_B = val
	low_B = min(high_B-1, low_B)
	cv.setTrackbarPos(low_B_name, window_detection_name, low_B)
def on_high_B_thresh_trackbar(val):
	global low_B
	global high_B
	high_B = val
	high_B = max(high_B, low_B+1)
	cv.setTrackbarPos(high_B_name, window_detection_name, high_B)
def on_low_G_thresh_trackbar(val):
	global low_G
	global high_G
	low_G = val
	low_G = min(high_G-1, low_G)
	cv.setTrackbarPos(low_G_name, window_detection_name, low_G)
def on_high_G_thresh_trackbar(val):
	global low_G
	global high_G
	high_G = val
	high_G = max(high_G, low_G+1)
	cv.setTrackbarPos(high_G_name, window_detection_name, high_G)
def on_low_R_thresh_trackbar(val):
	global low_R
	global high_R
	low_R = val
	low_R = min(high_R-1, low_R)
	cv.setTrackbarPos(low_R_name, window_detection_name, low_R)
def on_high_R_thresh_trackbar(val):
	global low_R
	global high_R
	high_R = val
	high_R = max(high_R, low_R+1)
	cv.setTrackbarPos(high_R_name, window_detection_name, high_R)

cv.namedWindow(window_detection_name)

cv.createTrackbar(low_B_name, window_detection_name , low_B, max_value, on_low_B_thresh_trackbar)
#cv.createTrackbar(high_B_name, window_detection_name , high_B, max_value_H, on_high_B_thresh_trackbar)
cv.createTrackbar(low_G_name, window_detection_name , low_G, max_value, on_low_G_thresh_trackbar)
#cv.createTrackbar(high_G_name, window_detection_name , high_G, max_value, on_high_G_thresh_trackbar)
cv.createTrackbar(low_R_name, window_detection_name , low_R, max_value, on_low_R_thresh_trackbar)
#cv.createTrackbar(high_R_name, window_detection_name , high_R, max_value, on_high_R_thresh_trackbar)

cap = cv.VideoCapture(args[1])

#man/min of past ten frames; average or total
def aggregate_rescaling(frame): #you only pca once
	global only_once
	global weights
	global max_min
	frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
	#frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	#kernel = np.ones((5,5),np.float32)/25
	#frame = cv.filter2D(frame,-1,kernel)

	r, c, d = frame.shape
	A = np.reshape(frame, (r * c, d))

	if not only_once:

		A_dot = A - A.mean(axis=0)[np.newaxis, :]

		_, eigv = LA.eigh(A_dot.T @ A_dot)
		weights = eigv[:, 0]
		#if (weights<0).sum() > 0:
		#if np.mean(weights) < 0:
		#	weights *= -1

		red = np.reshape(A_dot @ weights, (r, c))
		only_once = True
	else:
		red = np.reshape(A @ weights, (r, c))
	#red /= np.max(np.abs(red),axis=0) #this looks real cool - Damas
	"""
	if len(max_min['max']) == 10:
		max_min['max'] = max_min['max'][1:] + [np.max(red)]
		max_min['min'] = max_min['min'][1:] + [np.min(red)]
	else:
		max_min['max'].append(np.max(red))
		max_min['min'].append(np.min(red))
	"""

	if np.min(red) < max_min['min']:
		max_min['min'] = np.min(red)
	if np.max(red) > max_min['max']:
		max_min['max'] = np.max(red)

	#print(np.min(red), np.max(red), 'all time Domas', max_min['min'], max_min['max'])

	red -= max_min['min']
	red *= (255.0/(max_min['max'] - max_min['min']))

	#red -= np.min(max_min['min'])
	#red *= (255.0/np.abs(np.max(max_min['max'])))

	#red -= np.min(red)
	#red *= (255.0/np.abs(np.max(red)))
	if not paused:
		print(np.min(red), np.max(red), max_min['min'], max_min['max'])

	red = red.astype(np.uint8)
	red = np.expand_dims(red, axis = 2)
	red = np.concatenate((red, red, red), axis = 2)
	
	cv.imshow('One Time PCA plus all time aggregate rescaling', red)
	cv.imshow('frame', frame_gray)

paused = False
only_once = False
weights = []
max_min = {'max': 90, 'min': -20}
speed = 1
while True:
	if not paused:
		for _ in range(speed):
			ret, frame = cap.read()
	if ret:
		aggregate_rescaling(frame)
		#break
	key = cv.waitKey(30)
	if key == ord('q') or key == 27:
		break
	if key == ord('p'):
		paused = not paused
	if key == ord('i') and speed > 1:
		speed -= 1
	if key == ord('o'):
		speed += 1