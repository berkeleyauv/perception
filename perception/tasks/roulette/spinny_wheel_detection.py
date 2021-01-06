import numpy as np
import cv2
import argparse
import sys
import time

# TODO: port to vis + TaskPerciever format or remove

# CHANGE PARAMETER IN CALL TO "THRESH" FUNCTION TO CHANGE COLOR
file_name = "GOPR1145.MP4"  # video file from dropbox
vid = cv2.VideoCapture(file_name)
frames = 0
avgLength = 10
centers = []


def thresh(frame, color='red'):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if color == 'red':
        lower = np.uint8([29, 77, 36])
        upper = np.uint8([130, 250, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mask)
    elif color == 'blue':
        lower = np.uint8([86, 141, 0])
        upper = np.uint8([106, 220, 168])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower = np.uint8([66, 208, 157])
        upper = np.uint8([86, 255, 209])
        mask = cv2.inRange(hsv, lower, upper)

    return mask


def heuristic(contour):
    rect = cv2.minAreaRect(contour)
    area = rect[1][0] * rect[1][1]
    diff = cv2.contourArea(cv2.convexHull(contour)) - cv2.contourArea(contour)
    cent = rect[0]
    dist = 0
    if len(likelySection) > 1 and allLarger(60):
        cen0 = cv2.minAreaRect(likelySection[0]['cont'])[0]
        dis0 = np.linalg.norm(np.array(cent) - np.array(cen0))
        cen1 = cv2.minAreaRect(likelySection[1]['cont'])[0]
        dis1 = np.linalg.norm(np.array(cent) - np.array(cen1))
        dist = min([dis0, dis1])
    heur = area - 3 * diff - 20 * dist
    return heur


def allLarger(thresh):
    for cnt in likelySection:
        if cnt['heur'] < thresh:
            return False
    return True


def drawRects(frame, contours):
    tempPts = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt['cont'])
        boxpts = cv2.boxPoints(rect)
        box = np.int0(boxpts)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 1)
        cv2.drawContours(frame, [cnt['cont']], 0, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(cnt['cont'])], 0, (255, 0, 0), 1)
        tempPts.append(rect[0])
        # cv2.putText(frame, str(cnt['heur']), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    if len(tempPts) > 1 and allLarger(60):
        cv2.circle(frame, (int(tempPts[0][0]), int(tempPts[0][1])), 10, (0, 0, 255), -1)
        cv2.circle(frame, (int(tempPts[1][0]), int(tempPts[1][1])), 10, (0, 0, 255), -1)


def midPt(pt1, pt2):
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)


def getAvgPt(pt):
    points.append(pt)
    exes = list(map(lambda x: x[0], points))
    whys = list(map(lambda y: y[1], points))

    if len(points) > 50:
        del points[:10]
    return (int(sum(exes) / len(exes)), int(sum(whys) / len(whys)))


likelySection = []
points = []
while vid.isOpened():
    start = time.time()
    ret, frame = vid.read()
    if (ret == False):
        continue

    threshed = thresh(frame, 'other')
    res, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=heuristic, reverse=True)
    if len(contours) > 1:
        c1 = contours[0]
        c2 = contours[1]
        heur0 = heuristic(c1)
        heur1 = heuristic(c2)
        likelySection = [{'cont': contours[0], 'heur': heur0}, {'cont': contours[1], 'heur': heur1}]
    untampered = np.copy(frame)
    if contours:
        drawRects(frame, likelySection)
    cv2.imshow("Frame", frame)
    cv2.imshow('Res', res)

    if (cv2.waitKey(1) & 0xFF) == ord('q') or frames > 900:
        break

vid.release()
cv2.destroyAllWindows()
