import numpy as np
import cv2
import sys

#############################################################################
# Compilation of ways to do cross detection for this year's challenge.
# Add more functions for alternative algorithms!
#############################################################################

sys.path.insert(0, '../background_removal')
from perception.tasks.segmentation.peak_removal_adaptive_thresholding import filter_out_highest_peak_multidim
from perception.tasks.segmentation.combinedFilter import init_combined_filter

ret, frame = True, cv2.imread('../data/cross/cross.png')  # https://i.imgur.com/rjv1Vcy.png

# "hsv" = Apply hsv thresholding before trying to find the path marker
# "multidim" = Apply filter_out_highest_peak_multidim
# "combined" = Apply pca then multidim
thresholding = "combined" # Apply hsv thresholding before trying to find the path marker

def find_cross(frame, draw_figs=True):
    """ Returns the middle of a possible cross that has the largest contour area 

        One of the ideas from: 
        https://stackoverflow.com/questions/14612192/detecting-a-cross-in-an-image-with-opencv
    """

    # Or any other colorspace transformation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255,0)
    __, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    possible_crosses = []

    for i in range(len(contours)):
        cnt = contours[i]

        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(contours[i],hull)

        # Crosses have 4 "defects" (concave places)
        if defects is not None and len(defects) == 4:
            possible_crosses.append(defects)

    if draw_figs:
        img = frame.copy()
        for defects in possible_crosses:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # start = tuple(cnt[s][0])
                # end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                # cv2.line(img,start,end,[0,255,0],2)
                cv2.circle(img, far, 5, [0, 0, 255], -1)
            cv2.imshow('cross at contour number ' + str(i),img)
        cv2.imshow('original', frame)


    return possible_crosses[0]

###########################################
# Main Body
###########################################
# TODO: port to vis
if __name__ == "__main__":
    combined_filter = init_combined_filter()

    ret_tries = 0
    while 1 and ret_tries < 50:
        # ret,frame = cap.read()
        if ret:
            # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            if thresholding == "multidim":
                votes1, threshed = filter_out_highest_peak_multidim(frame)
                threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            elif thresholding == "combined":
                threshed = combined_filter(frame)
            else:
                threshed = frame.copy()

            cross_pts = find_cross(frame, True)
            input() # This test is only for one frame

            ret_tries = 0
            k = cv2.waitKey(60) & 0xff
            if k == 27:  # esc
                break
        else:
            ret_tries += 1

    cv2.destroyAllWindows()
