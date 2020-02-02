import numpy as np
import cv2

# Get input from webcam
cap = cv2.VideoCapture(0)
def nothing(x):
	pass
cap = cv2.VideoCapture(0)
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


# Continue until user ends program
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

    cv2.imshow('nine', np.vstack((np.hstack((res, res, res)), np.hstack((res, res, res)), np.hstack((res, res, res)))))


    if cv2.waitKey(1) and 0xFF == ord('q'): # Exit
        break



#Cleanup
cap.release()
cv2.destroyAllWindows()
