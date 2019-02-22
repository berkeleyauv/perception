import numpy as np
import cv2
import sys

#### TODO: maybe look into pattern matching

# Data fron the new course footage dropbox folder
cap = cv2.VideoCapture('../data/course_footage/path_marker_GOPR1142.mp4')
path_marker_template = cv2.imread('../data/patterns/path_marker_tip.png')

##################################################################################
#
# Code in this region is *similar* to that in play_roulette_detection.py as of 2/13/2019
#
#

testing = False # Show hsv sliders and threshold image.

# roulette hole: 0.02 and 0.005
# These extreme values disable hsv adaptive thresholding
hsv_thresh_high = 100
hsv_thresh_low = 0

# HSV threshold values. Can be changed during runtime if testing
# Use python3 <filename.py> test to open testing mode

# path marker
h_low = 31
s_low = 28
v_low = 179
h_hi = 79
s_hi = 88
v_hi = 218

thresholds_used = [h_low, s_low, v_low, h_hi, s_hi, v_hi]

def nothing(x):
    """Helper method for the trackbar"""
    pass

def init_test_hsv_thresholds(thresholds):
    # Keep track of previous threhold values to see if the user is using the trackbar
    # is there a function that detects whether the mouse button is down?
    prev_h_low, prev_s_low, prev_v_low, prev_h_hi, prev_s_hi, prev_v_hi = thresholds

    def test_hsv_thresholds(frame, thresholds):
        nonlocal prev_h_low, prev_s_low, prev_v_low, prev_h_hi, prev_s_hi, prev_v_hi

        h_low_track = cv2.getTrackbarPos('h_low','contours')
        s_low_track = cv2.getTrackbarPos('s_low','contours')
        v_low_track = cv2.getTrackbarPos('v_low','contours')
        h_hi_track = cv2.getTrackbarPos('h_high','contours')
        s_hi_track = cv2.getTrackbarPos('s_high','contours')
        v_hi_track = cv2.getTrackbarPos('v_high','contours')

        if h_low_track!=prev_h_low or s_low_track!=prev_s_low or v_low_track!=prev_v_low \
                or h_hi_track!=prev_h_hi or s_hi_track!=prev_s_hi or v_hi_track!=prev_v_hi:
            # If user is adjusting the trackbars, use the user input
            thresholds_used = [h_low_track, s_low_track, v_low_track, h_hi_track, s_hi_track, v_hi_track]
        else:
            # Otherwise, copy program data to trackbars
            thresholds_used = thresholds
            cv2.setTrackbarPos('h_low', 'contours', thresholds_used[0])
            cv2.setTrackbarPos('s_low', 'contours', thresholds_used[1])
            cv2.setTrackbarPos('v_low', 'contours', thresholds_used[2])
            cv2.setTrackbarPos('h_high', 'contours', thresholds_used[3])
            cv2.setTrackbarPos('s_high', 'contours', thresholds_used[4])
            cv2.setTrackbarPos('v_high', 'contours', thresholds_used[5])

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(thresholds_used[:3]), np.array(thresholds_used[3:]))
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('contours', res)

        prev_h_low, prev_s_low, prev_v_low, prev_h_hi, prev_s_hi, prev_v_hi = thresholds_used
        return thresholds_used

    return test_hsv_thresholds

def hsv_threshold(frame, thresh_used, tries=0):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(thresh_used[:3]), np.array(thresh_used[3:]))
    res = cv2.bitwise_and(frame,frame, mask= mask)

    if tries < 3:
        if np.count_nonzero(res) > res.shape[0]*res.shape[1] * hsv_thresh_high:
            # narrow the threshold and retry
            thresh_used[0] += 1
            thresh_used, res = hsv_threshold(frame, thresh_used, tries+1)
        if np.count_nonzero(res) < res.shape[0]*res.shape[1] * hsv_thresh_low:
            # widen the threshold and retry
            thresh_used[0] -= 1
            thresh_used, res = hsv_threshold(frame, thresh_used, tries+1)
    return thresh_used, res

def filter_for_rectangles(contours):
    rects = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4 or len(approx) == 8:
            rects.append(c)
    return rects

def close_to(rect1, rect2, threshold):
    ### returns whether rect1 is close to rect2 based on threshold
    if rect1 is None:
        return True
    dx, dy = rect1[0][0]-rect2[0][0], rect1[0][1]-rect2[0][1];
    return dx**2 + dy**2 < threshold * threshold

def init_tracker(tracker_num):
    # tracker_types: ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if tracker_num == 1:
        tracker = cv2.TrackerBoosting_create()
    elif tracker_num == 2:
        tracker = cv2.TrackerMIL_create()
    elif tracker_num == 3:
        tracker = cv2.TrackerKCF_create()
    elif tracker_num == 4:
        tracker = cv2.TrackerTLD_create()
    elif tracker_num == 5:
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_num == 6:
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_num == 7:
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_num == 8:
        tracker = cv2.TrackerCSRT_create()
    else:
        print("Invalid tracker number")
        exit()
    return tracker
#
#
##################################################################################

def line_length(line):
    x0,y0,x1,y1 = line[0]
    return (x0-x1)**2 + (y0-y1)**2

def find_path_marker(frame):
    """ Returns angle of bottom line and top line relative to 0 radians
        This function doesn't guarantee that the angles are distinct
        Returns None if no good lines are found """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)

    # Find Hough lines
    # Source: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180 # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 2  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    line_image = frame.copy()
    if lines is not None:
        lines = lines.tolist()
        lines.sort(key=line_length, reverse=True)
        lines = lines[:10]

        # if line's y0 is below the average y0, it is a part of the top line, opposite for bottom line
        avgy = sum([l[0][1] for l in lines]) // len(lines)
        bot_lines = [l for l in lines if l[0][1] > avgy]
        top_lines = [l for l in lines if l[0][1] < avgy]

        for l in bot_lines:
            line_image = cv2.line(line_image, tuple(l[0][0:2]), tuple(l[0][2:4]), (0, 255, 0), 5)
        for l in top_lines:
            line_image = cv2.line(line_image, tuple(l[0][0:2]), tuple(l[0][2:4]), (255, 0, 0), 5)

        if len(bot_lines) > 0 and len(top_lines) > 0:
            # Sometimes, these two angles are the same :c
            bot_angle = sum([np.arctan2(l[0][1]-l[0][3],l[0][0]-l[0][2]) for l in bot_lines]) / len (bot_lines)
            top_angle = sum([np.arctan2(l[0][1]-l[0][3],l[0][0]-l[0][2]) for l in top_lines]) / len (top_lines)

            line_image = draw_marker_angles(line_image, (bot_angle, top_angle))
            cv2.imshow('lines', line_image)

            return bot_angle, top_angle
        else:
            cv2.imshow('lines', line_image)

            return None

def draw_marker_angles(frame, marker_angles):
    """ Draws lines with the same angles as those in marker_angles off to
        the side of the frame """

    line_image = frame.copy()
    bot_angle, top_angle = marker_angles

    h, w = frame.shape[:2]
    x, y = w*0.75, h*0.5
    r = 20
    pt_mid = (int(x), int(y))
    pt_bot = (int(x + r*np.cos(bot_angle)), int(y + r*np.sin(bot_angle)))
    pt_top = (int(x + r*np.cos(top_angle)), int(y + r*np.sin(top_angle)))
    line_image = cv2.line(line_image, pt_mid, pt_bot, (0, 0, 255), 5)
    line_image = cv2.line(line_image, pt_mid, pt_top, (0, 0, 255), 5)

    return line_image

###########################################
# Main Body
###########################################

if __name__ == "__main__":
    if len(sys.argv) > 0:
        if "test" in sys.argv:
            testing = True
            cv2.namedWindow('contours')
            cv2.createTrackbar('h_low','contours',h_low,255,nothing)
            cv2.createTrackbar('s_low','contours',s_low,255,nothing)
            cv2.createTrackbar('v_low','contours',v_low,255,nothing)
            cv2.createTrackbar('h_high','contours',h_hi,255,nothing)
            cv2.createTrackbar('s_high','contours',s_hi,255,nothing)
            cv2.createTrackbar('v_high','contours',v_hi,255,nothing)

    test_hsv_thresholds = init_test_hsv_thresholds(thresholds_used)
    marker_angles = None

    # For testing purposes
    for _ in range(50):
        cap.read()

    while(1):
        ret,frame = cap.read()

        if ret == True:
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            if testing:
                thresholds_used = test_hsv_thresholds(frame, thresholds_used)

            thresholds_used, hsv_thresh = hsv_threshold(frame, thresholds_used)
            new_marker_angles = find_path_marker(hsv_thresh)
            if new_marker_angles is not None:
                marker_angles = new_marker_angles

            # draw marker angles onto original image
            marker_img = frame.copy()
            if marker_angles != None:
                marker_img = draw_marker_angles(frame, marker_angles)
            cv2.imshow("roulette hole", marker_img)

        k = cv2.waitKey(60) & 0xff
        if k == 27: # esc
            if testing:
                print("hsv thresholds:")
                print(thresholds_used)
            break

    cv2.destroyAllWindows()
    cap.release()