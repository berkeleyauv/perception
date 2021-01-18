import numpy as np
import cv2
import sys

# TODO: port to vis + TaskPerciever format or remove

#### TODO: maybe look into pattern matching

# Data fron the new course footage dropbox folder
cap = cv2.VideoCapture('../data/course_footage/play_slots_GOPR1142.mp4')

detecting = True # Use hsv thresholding and rectangle detection. Always True
tracking = False # Use trackers to try to accomodate for failure
tracker_num = 5 # The tracker to use
testing = False # Show hsv sliders and threshold image.

detection_interval = 10 # If tracking
fail_thresh = 5 # Number of detection failures before rectangle disappears from output
close_thresh = 30 # Pixel threshold used to reject dissimilar consecutive detection results
slots_size_thresh = 50 # Minimum area of detected slots slot
slots_dimension_thresh = 0.3 # Slot result's maximum % deviation from expected width height ratio

# HSV threshold values. Can be changed during runtime if testing
# Use python3 play_slots_detection test to open testing mode
h_low = 96
s_low = 82
v_low = 131
h_hi = 190
s_hi = 180
v_hi = 228

def nothing(x):
    """Helper method for the trackbar"""
    pass

def test_hsv_thresholds(frame, has_input=False,h_low=None,s_low=None,v_low=None,h_hi=None,s_hi=None,v_hi=None):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    if has_input:
        cv2.setTrackbarPos('h_low', 'contours', h_low)
        cv2.setTrackbarPos('s_low', 'contours', s_low)
        cv2.setTrackbarPos('v_low', 'contours', v_low)
        cv2.setTrackbarPos('h_high', 'contours', h_hi)
        cv2.setTrackbarPos('s_high', 'contours', s_hi)
        cv2.setTrackbarPos('v_high', 'contours', v_hi)
    
    h_low = cv2.getTrackbarPos('h_low','contours')
    s_low = cv2.getTrackbarPos('s_low','contours')
    v_low = cv2.getTrackbarPos('v_low','contours')
    h_hi = cv2.getTrackbarPos('h_high','contours')
    s_hi = cv2.getTrackbarPos('s_high','contours')
    v_hi = cv2.getTrackbarPos('v_high','contours')

    mask = cv2.inRange(hsv, np.array([h_low,s_low,v_low]), np.array([h_hi,s_hi,v_hi]))
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('contours', res)

    return h_low, s_low, v_low, h_hi, s_hi, v_hi

def hsv_threshold(frame, _h_low, s_low, v_low, h_hi, s_hi, v_hi, tries=0):
    global h_low
    h_low = _h_low
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_low,s_low,v_low]), np.array([h_hi,s_hi,v_hi]))
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Threshold depend on whether the sub is close to or far from the target
    if tries < 3:
        if np.count_nonzero(res) > res.shape[0]*res.shape[1] * 0.02:
            # narrow the threshold and retry
            h_low += 1
            res = hsv_threshold(frame, h_low, s_low, v_low, h_hi, s_hi, v_hi, tries+1)
        if np.count_nonzero(res) < res.shape[0]*res.shape[1] * 0.005:
            # widen the threshold and retry
            h_low -= 1
            res = hsv_threshold(frame, h_low, s_low, v_low, h_hi, s_hi, v_hi, tries+1)
    return res

def filter_for_rectangles(contours):
    rects = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4 or len(approx) == 8:
            rects.append(c)
    return rects

def find_red_slots_hole(frame, size_thresh, dimension_thresh):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)

    im, contours, hierarchy = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    def get_area(rect):
        return rect[1][0] * rect[1][1]
    def wh_ratio(rect):
        return max(rect[1])/min(rect[1])
    def dim_ratio(rect, reference):
        return abs(reference-wh_ratio(rect))/reference
    def is_open(h):
        return h[2] < 0

    # contours = filter_for_rectangles(contours) # Makes it worse right now lol

    # Take the first few contours and find the one that fits the dimensions the best
    # The play slots rectangle is square
    contours = [cv2.minAreaRect(c) for c in contours]
    contours = [c for c in contours if get_area(c) > size_thresh]

    if len(contours) > 0:
        contours = [c for c,h in zip(contours, hierarchy[0]) if dim_ratio(c,1/1)<dimension_thresh 
                or is_open(h)]
        contours.sort(key=lambda c: get_area(c), reverse=True)

    if len(contours) > 0:
        return contours[0]
    else:
        return None

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

    tracker = init_tracker(tracker_num)
    slots_hole = None
    num_failures = 0
    time_since_detection = 0

    # # For testing purposes
    # for _ in range(500):
    #     cap.read()

    while(1):
        ret ,frame = cap.read()

        if ret == True:
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            if testing:
                h_low, s_low, v_low, h_hi, s_hi, v_hi = test_hsv_thresholds(frame,True,h_low,s_low,v_low,h_hi,s_hi,v_hi)

            if time_since_detection >= detection_interval:
                detecting = True

            if detecting and tracking:
                hsv_thresh = hsv_threshold(frame, h_low, s_low, v_low, h_hi, s_hi, v_hi)
                new_slots_hole = find_red_slots_hole(hsv_thresh, slots_size_thresh, slots_dimension_thresh)
                if new_slots_hole is not None and close_to(slots_hole, new_slots_hole, close_thresh):
                    num_failures = 0
                    slots_hole = new_slots_hole
                    tracker = init_tracker(tracker_num)# does this have to happen every time?
                    tracker.init(frame, slots_hole[0]+slots_hole[1])
                    detecting = False
                    time_since_detection = 0
                else:
                    num_failures += 1
            elif detecting and not tracking:
                hsv_thresh = hsv_threshold(frame, h_low, s_low, v_low, h_hi, s_hi, v_hi)
                new_slots_hole = find_red_slots_hole(hsv_thresh, slots_size_thresh, slots_dimension_thresh)
                if new_slots_hole is not None and close_to(slots_hole, new_slots_hole, close_thresh):
                    num_failures = 0
                    slots_hole = new_slots_hole
                else:
                    num_failures += 1
            elif tracking:
                ret, bounding_box = tracker.update(frame)
                if ret:
                    num_failures = 0
                    slots_hole = (bounding_box[:2], bounding_box[2:4], slots_hole[2])
                else:
                    num_failures += 1
            if num_failures > fail_thresh:
                slots_hole = None
                detecting = True

            # draw slots hole onto original image
            slots_img = frame.copy()
            if slots_hole != None:
                box = np.int0(cv2.boxPoints(slots_hole))
                slots_img = cv2.drawContours(slots_img, [box], 0, (0,255,0), 2)
            cv2.imshow("slots hole", slots_img)


            time_since_detection += 1
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                if testing:
                    print("hsv thresholds:")
                    print(h_low, s_low, v_low, h_hi, s_hi, v_hi)
                break

    cv2.destroyAllWindows()
    cap.release()
