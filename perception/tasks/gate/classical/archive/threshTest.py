from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

# TODO: port to vis + TaskPerciever format or remove

#expectations
#contours closest to the last ones
#should know when we passed through the gate
"""
IMPORTANT!!!! RUN THIS WITH $ python3 threshTest.py GOPR1142.mp4
"""
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 98
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
pauseWhenFound = 0
old_gray = None
p0 = None
heur_thresh = 200

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

def trueFalsePause(val):
    global pauseWhenFound
    pauseWhenFound = val
    cv.setTrackbarPos('pausing', window_capture_name, pauseWhenFound)
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def drawRects(frame, contours):
    tempPts = []
    for cnt in contours:
        rect = cv.minAreaRect(cnt['cont'])
        boxpts = cv.boxPoints(rect)
        box = np.int0(boxpts)
        cv.drawContours(frame,[box],0,(0,0,255),1)
        cv.drawContours(frame, [cnt['cont']],0,(0,255,0),1)
        cv.drawContours(frame, [cv.convexHull(cnt['cont'])],0,(255,0,0),1)
        tempPts.append(rect[0])
        cv.putText(frame, str(cnt['heur']), (int(rect[0][0]), int(rect[0][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    if len(tempPts) > 1 and allLarger(heur_thresh):
        global paused
        if pauseWhenFound:
            paused = True
        avgPt = getAvgPt(midPt(tempPts[0], tempPts[1]))
        cv.circle(frame, (avgPt[0], avgPt[1]), 10, (0,0,255), -1)

def midPt(pt1, pt2):
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

def getAvgPt(pt):
    points.append(pt)
    exes = list(map(lambda x: x[0], points))
    whys = list(map(lambda y: y[1], points))

    if len(points) > 50:
        del points[:10]
    return (int(sum(exes) / len(exes)), int(sum(whys) / len(whys)))

def heuristic(contour):
    rect = cv.minAreaRect(contour)
    area = rect[1][0] * rect[1][1]
    diff = cv.contourArea(cv.convexHull(contour)) - cv.contourArea(contour)
    cent = rect[0]
    dist = 0
    if len(likelyGate) > 1 and allLarger(heur_thresh):
        cen0 = cv.minAreaRect(likelyGate[0]['cont'])[0]
        dis0 = np.linalg.norm(np.array(cent) - np.array(cen0))
        cen1 = cv.minAreaRect(likelyGate[1]['cont'])[0]
        dis1 = np.linalg.norm(np.array(cent) - np.array(cen1))
        dist = min([dis0, dis1])
    heur = area - 3 * diff - 20 * dist #only factor in dist with all heurs larger than 60
    #print(heur)
    return heur

def allLarger(thresh):
    for cnt in likelyGate:
        if cnt['heur'] < thresh:
            return False
    return True

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('camera', help='Camera devide number.', default=0, type=str)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)

cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
cv.createTrackbar('pausing', window_capture_name, pauseWhenFound, 1, trueFalsePause)

#cv.createTrackbar('low_canny', 'canny', low_canny, 500, lcanny)
paused = False

likelyGate = []
points = []
while True:
    if not paused:
        ret, frame = cap.read() #reads the frame
    else:
        frame = untampered
    if ret:
        if not paused:
            frame = cv.resize(frame, (0,0), fx=0.4, fy=0.4)#resizes frame so that it fits on screen
            blur = cv.GaussianBlur(frame, (5, 5), 0)
            frame_HSV = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
            #frame_gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
            #canny = cv.Canny(frame_gray, 200, 3, True)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V)) #low_S ideal = 98 Sets threshold in hsv
    
        frame_threshold = cv.bitwise_not(frame_threshold)
        res = cv.bitwise_and(frame,frame, mask= frame_threshold)
        res2, contours, hierarchy = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contours.sort(key=heuristic, reverse=True) #sorts the list of contours by a heuristic function, based on area, distance from previous contours
        if len(contours) > 1:#two largest heuristics are assumed to be the two gate posts
            heur0 = heuristic(contours[0])
            heur1 = heuristic(contours[1])
            likelyGate = [{'cont': contours[0], 'heur': heur0}, {'cont': contours[1], 'heur': heur1}]

        """
        ### This is very crude adaptive thresholding
        totArea = 0
        for cnt in contours:
            totArea += cv.contourArea(cnt)
        if totArea > 19000:
            low_S -= .5
        if low_S < 250:
            low_S += .2
        ###
        """

        untampered = np.copy(frame)
        #cv.putText(frame, str(low_S)+' '+str(totArea), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        if contours:
            #likelyGate.append(contours[0])
            #findLikelyGate(likelyGate, contours)
            drawRects(frame, likelyGate)
        cv.imshow(window_capture_name, frame)
        cv.imshow(window_detection_name, frame_threshold)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
    if key == ord('p'):
        paused = not paused

#generalized problem, giving center of object contrasting with water