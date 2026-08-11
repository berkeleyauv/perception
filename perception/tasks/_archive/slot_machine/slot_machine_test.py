import numpy as np
import cv2 as cv
from utils import *

def get_colors(frame, contour, x,y):
    shaped = np.float32(frame.reshape(-1,3))
    x_curr = x[0]
    y_curr = y[0]
    # changed = []
    # for shape in shaped:
    #     if (cv.pointPolygonTest(contour,(x_curr,y_curr),False) >= 0):
    #         changed.append(shape)
    #     x_curr+=1
    #     if(x_curr-1 == x[1]):
    #         # print(y_curr)
    #         x_curr = x[0]
    #         y_curr+=1
    # changed = np.float32(np.array(changed))
    # shaped = [shape for shape in shaped if cv.pointPolygonTest(contour,shape,False) >= 0]
    n_colors = 2
    # print(changed,shaped)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS
    # print(len(shaped)-len(changed))
    # try:
    #     _, labels, palette = cv.kmeans(changed, n_colors, None, criteria, 10, flags)
    # except:
    _, labels, palette = cv.kmeans(shaped, n_colors, None, criteria, 10, flags)
    
    _, counts = np.unique(labels, return_counts=True)
    ret = tuple(palette[np.argmax(counts)])
    ret = ( int (ret [ 0 ]), int (ret [ 1 ]), int (ret [ 2 ])) 
    # print(ret)
    return ret

def resize_frame(frame,ratio = 0.4):
    return cv.resize(frame,(int(frame.shape[1]*ratio),int(frame.shape[0]*ratio)))

cap = cv.VideoCapture("./water_spinner.MP4")
while not cap.isOpened():
    cap = cv.VideoCapture("./water_spinner.MP4")
    cv.waitKey(1000)
    print ("Wait for the header")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = resize_frame(frame)
        orig = frame

        bw_frame = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
        high_thresh, thresh_im = cv.threshold((bw_frame), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow('thres',thresh_im)
        lowThresh = 0.5*high_thresh
        # print(high_thresh, lowThresh)
        R,G,B = cv.split(frame)
        # smooth = cv.bilateralFilter(frame,10,25,51)
        smooth = cv.blur(frame,(2,2))
        # cv.imshow('test',smooth)

        # GENERATE CANNY EDGES
        canny = cv.Canny(smooth, 100, high_thresh)

        sigma = 0.6
        v = np.median(smooth)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        # print(lower, upper)
        # canny = cv.Canny(smooth, lower, upper)
        #TODO: ADD TO RET FRAMES
        # cv.imshow('canny_new',canny)

        ## FIND CONTOURS
        # _, contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        outer_shell = np.zeros(frame.shape, np.uint8)
        #keep only contours of more or less correct area and perimeter
        cv.drawContours(outer_shell,contours,-1,(255,255,0),1)
        cv.imshow('shell',outer_shell)

        w, h,c  = frame.shape
        blank = np.zeros((w, h)).astype(np.uint8)
        cv.drawContours(blank, contours, -1, 1, 1)
        blank = cv.morphologyEx(blank, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, contours, _ = cv.findContours(blank, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(frame.shape, np.uint8)
        #keep only contours of more or less correct area and perimeter
        contours = [c for c in contours if 100 < cv.contourArea(c) < 800]
        cv.drawContours(mask,contours,-1,(255,255,0),1)
        for i,c in enumerate(contours):
            # mask = np.zeros(frame.shape, np.uint8)
            # cv.drawContours(frame, c, -1,(0,0,255),1)
            x,y,w,h = cv.boundingRect(c) # offsets - with this you get 'mask'
            cimg = np.zeros_like(smooth)
            # cv.imshow('cutted contour',frame[y:y+h,x:x+w])
            pts = cimg
            # print(frame[c[1][0]])   
            # arr = []
            # for v in c:
            #     try:
            #         arr.append(frame[v[0]])         
            #     except:
            #         pass
            # arr = np.array(arr)
            # np.array([frame[v[0]] for v in c])
            # get_colors(frame[y:y+h,x:x+w])
            # if 0.4 <= h/w <= 1.3 and 10 < len(cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)) and (cv.contourArea(c)/cv.arcLength(c, True)) > 0.5:
            cv.rectangle(frame,(x,y),(x+w,y+h),get_colors(smooth[y:y+h,x:x+w],c,(y,y+h),(x,x+w)),2)
        cv.imshow('res',frame)
        # cv.imshow('eql',eql)

        # MASK TO SEE EDGES

        hulls = []
        usec = []
        for c in contours:
            x,y,w,h = cv.boundingRect(c)
            # if 10 < len(cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)) and (cv.contourArea(c)/cv.arcLength(c, True)) > 0.5:
            hull = cv.convexHull(c)
            # if cv.contourArea(hull)/cv.contourArea(c) > 0.8:
            hulls.append(hull)
                # cv.fillPoly(thresh_im, pts=[c], color=0)
            usec.append(c)
                # cv.drawContours(mask, c, -1, (0,255,0),1)
        # top = sorted(list(zip(hulls,usec)),key= lambda h: cv.contourArea(h[0]) - cv.contourArea(h[1]))
        # print(top)
        # arr = np.array(top)
        cv.drawContours(mask,hulls,-1,(255,0,0),1)
        cv.drawContours(mask,usec,-1,(0,255,0),1)
        cv.imshow('cont',mask)


        # cv.imshow('orig',orig)
        # cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(32) == ord(' '):
            while(not cv.waitKey(32) == ord(' ')):
                continue

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()