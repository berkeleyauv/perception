import numpy as np
import cv2
import argparse
import sys
from PIL import Image
import time

video_file = 'truncated_semi_final_run.mp4'
EPSILON = 40
OVERLAP_EPS = 40

class Contour:
	def __init__(self, _x, _y, _w, _h, _area):
		self.x = _x
		self.y = _y
		self.w = _w
		self.h = _h
		self.area = _area

	def __str__(self):
		return str(self.__dict__)

	def __eq__(self, other):
		return self.__dict__ == other.__dict__

def imgDetect(file):
	if isinstance(file, str):
		frame = cv2.imread(file)
	else:
		frame = file

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

	lower = np.array([29,40,36], dtype='uint8')
	upper = np.array([77,80,50], dtype='uint8')
	mask = cv2.inRange(hsv, lower, upper)
	#filtered = cv2.bitwise_and(frame, frame, mask=mask)
	blur = cv2.GaussianBlur(gray, (3,3), 2)
	#thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
	canny = cv2.Canny(blur, 10, 80)
	#img, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	img = frame.copy()
	myContours = getContours(canny)
	sortedContours = sorted(myContours, key=lambda x: x.area)
	if len(sortedContours) > 1:
		a,b = sortedContours[-1], sortedContours[-2]
		x,y = (a.x+b.x+a.w//2+b.w//2)//2, (a.y+b.y+a.h//2+b.h//2)//2
		cv2.circle(img, (x,y), radius=5, color=(0,255,0), thickness=2)
		#cv2.putText(img, "Center Point", (x,y-10), 2, 0.5, (0,0,0))
	# for cnt in sortedContours[:-2]:
	# 	cv2.rectangle(img, (cnt.x, cnt.y), (cnt.x+cnt.w, cnt.y+cnt.h), (0,0,255), 2)
	for cnt in sortedContours[-2:]:
		cv2.rectangle(img, (cnt.x, cnt.y), (cnt.x+cnt.w, cnt.y+cnt.h), (255,0,0), 2)
	cv2.imshow('Canny', canny)
	if isinstance(file, str):
		cv2.imshow('Frame', frame)
		cv2.imshow('HSV', hsv)
		#cv2.imshow('Mask', mask)
		cv2.imshow('Canny', canny)
		cv2.imshow('Output', img)
		cv2.waitKey(0)
	return img

def videoDetect(file):
	vid = cv2.VideoCapture(file)
	frames = 0
	FPS = 30
	saver = cv2.VideoWriter('gateDetectionVideo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (640,360))
	while vid.isOpened():
		start = time.time()
		ret, frame = vid.read()
		img = imgDetect(frame)
		frames += 1
		# print(frames)
		# if frames == 500:
		# 	cv2.imwrite('test.png', frame)
		cv2.imshow('Frame', frame)
		cv2.imshow('Output', img)
		saver.write(img)
		end = time.time()
		if (cv2.waitKey(1) & 0xFF) == ord('q') or frames > 900:
			break
	vid.release()
	cv2.destroyAllWindows()

def getContours(image):
	start = time.time()
	# blur = cv2.bilateralFilter(cropped, 9, 17, 17)

	# edge = cv2.Canny(blur, 10, 100)
	# #cv2.imwrite('cannyContour.jpg', edge)
	# edge = cv2.GaussianBlur(edge, (5,5), 0.6)
	#cv2.imwrite('gaussianContour.jpg', edge)
	#Image.fromarray(edge).show()

	img, contours, hier = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	rectContour = []
	contourImg = image.copy()
	limited = image.copy()
	combined = image.copy()
	for cn in contours:
		#area = cv2.contourArea(cn)
		x,y,w,h = cv2.boundingRect(cn)
		area = w*h
		cv2.rectangle(contourImg, (x,y), (x+w, y+h), 1, 1)
		if area > 100 and area < 10000 and h>w+10:
			rectContour.append(Contour(x,y,w,h, cv2.contourArea(cn)))
			cv2.rectangle(limited, (x,y), (x+w, y+h), 1, 1)
	#cv2.imshow('Contours', blur)
	#cv2.imwrite('allContours.jpg', contourImg)
	#contourImg = Image.fromarray(contourImg)
	#contourImg.show()
	#cv2.imwrite('filteredContours.jpg', limited)
	#limited = Image.fromarray(limited)
	#limited.show()
	#combinedContours = combineTouchingContours(rectContour)
	#combinedContours = self.combineContours(rectContour)
	#combinedContours = self.combineContours(self.combineContours(rectContour))
	combinedContours = combineContours(combineContours(combineContours(rectContour)))

	for cn in combinedContours:
		x,y,w,h = cn.x, cn.y, cn.w, cn.h
		cv2.rectangle(combined, (x,y), (x+w, y+h), 1, 1)
	#cv2.imshow('Combined Contours', invert)
	cv2.imwrite('combinedContours.jpg', combined)
	#combined = Image.fromarray(combined)
	#combined.show()

	#print('Find Contours Time: ', time.time() - start)
	# for i in range(len(rectContour)):
	#     print('Contour ', i, ': ', str(rectContour[i]))

	return combinedContours

def combineContours(contours):
	newContours = []
	for cnt in contours:
		add = True
		for other in newContours:
			if cnt != other and Intersect(cnt, other):
				merged = Merge(cnt, other)
				if cnt in newContours:
					newContours.remove(cnt)
				newContours.remove(other)
				newContours.append(merged)
				add = False
		if add:
			newContours.append(cnt)
	return newContours

def Intersect(A, B):
	left = max(A.x, B.x)
	top = max(A.y, B.y)
	right = min(A.x + A.w, B.x + B.w)
	bottom = min(A.y + A.h, B.y + B.h)
	return (left <= right or abs(left-right) <= OVERLAP_EPS) and ((abs(A.y-B.y) <= EPSILON or abs(A.y+A.h-B.y-B.h) <= EPSILON)) and abs(A.y-B.y) <= EPSILON*2 and abs(A.y+A.h-B.y-B.h) <= EPSILON*2


def Merge(A, B):
	left = min(A.x, B.x)
	top = min(A.y, B.y)
	right = max(A.x + A.w, B.x + B.w)
	bottom = max(A.y + A.h, B.y + B.h)
	return Contour(left, top, right - left, bottom - top, A.area+B.area)


ap = argparse.ArgumentParser()
ap.add_argument('file_name', type=str, help='File name of video or image')
ap.add_argument('--test', '-t', action='store_true')

if __name__ == '__main__':
	args = ap.parse_args()
	if args.file_name.endswith('.mp4'):
		videoDetect(args.file_name)
	else:
		imgDetect(args.file_name)