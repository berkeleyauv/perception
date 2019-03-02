import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

########################################################################
# An attempt at an adaptive thresholding algorithm.
# Identifies "largest" peak in each of the three color channels
# of RBG and/or HSV images (others are possible) and then uses cv2.inRange() 
# to filter out background pixels.
#
# Program in action: https://i.imgur.com/FSfHGa6.png
# (Warning: this opens up a lot of windows)
# 
# Note: this uses functions from the scipy library (downloaded thru pip)
########################################################################


# cap = cv2.VideoCapture('../data/course_footage/GOPR1142.MP4')
# # No thresholds
# h_low = 0
# s_low = 0
# v_low = 0
# h_hi = 255
# s_hi = 255
# v_hi = 255

# cap = cv2.VideoCapture('../data/course_footage/path_marker_GOPR1142.mp4')
# # Path marker default
# h_low = 31
# s_low = 28
# v_low = 179
# h_hi = 79
# s_hi = 88
# v_hi = 218

cap = cv2.VideoCapture('../data/course_footage/play_slots_GOPR1142.MP4')
# Play slots default
h_low = 96
s_low = 82
v_low = 131
h_hi = 190
s_hi = 180
v_hi = 228

thresholds_used = [h_low, s_low, v_low, h_hi, s_hi, v_hi]

def init_test_hsv_thresholds(thresholds):
    # Keep track of previous threhold values to see if the user is using the trackbar
    # is there a function that detects whether the mouse button is down?
    prev_h_low, prev_s_low, prev_v_low, prev_h_hi, prev_s_hi, prev_v_hi = thresholds

    def nothing(x):
        """Helper method for the trackbar"""
        pass

    cv2.namedWindow('contours')
    cv2.createTrackbar('h_low','contours',h_low,255,nothing)
    cv2.createTrackbar('s_low','contours',s_low,255,nothing)
    cv2.createTrackbar('v_low','contours',v_low,255,nothing)
    cv2.createTrackbar('h_high','contours',h_hi,255,nothing)
    cv2.createTrackbar('s_high','contours',s_hi,255,nothing)
    cv2.createTrackbar('v_high','contours',v_hi,255,nothing)

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

def hsv_threshold(frame, thresh_used):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(thresh_used[:3]), np.array(thresh_used[3:]))
    res = cv2.bitwise_and(frame,frame, mask= mask)
    return thresh_used, res

def disp_hist(frame, title, labels, colors):
    frame0 = frame[:,:,0].flatten()
    frame0 = frame0[frame0 > 0]

    frame1 = frame[:,:,1].flatten()
    frame1 = frame1[frame1 > 0]

    frame2 = frame[:,:,2].flatten()
    frame2 = frame2[frame2 > 0]

    plt.figure(hash(title))
    plt.clf()
    ax = plt.gca()
    ax.set_xlim([0, 255])

    plt.hist(frame0, alpha=0.5, label=labels[0], color=colors[0])
    plt.hist(frame1, alpha=0.5, label=labels[1], color=colors[1])
    plt.hist(frame2, alpha=0.5, label=labels[2], color=colors[2])
    plt.title(title)
    plt.legend()
    plt.draw()

def find_peak_ranges(frame, display_plots=False, title=None, labels=None, colors=None):
    """ Finds a returns the widest peak's x-range in all three channels of frame 
        Result is formatted to fit cv2.inRange() -> ((low1, low2, low3), (hi1, hi2, hi3)) """

    # TODO: Maybe use a different combination of peak characteristics to more accurately
    #       select the entire peak (only the tip is selected right now)

    def find_widest_peak(channel, display_plots=False):
        """ Finds and returns the x-range of the widest peak in the
            given channel of frame """

        f = frame[:, :, channel].flatten()

        # Some semi-hardcoded values :)
        num_bins = max(int((np.amax(f)-np.amin(f)) / 4), 10)

        hist, bins = np.histogram(f, bins=num_bins)

        hist[0] = 0 # get rid of stuff that was thresholded to 0

        peaks, properties = find_peaks(hist, threshold=0.1, prominence=0.1, width=0.01)

        widths = peak_widths(hist, peaks)[0]
        i = np.argmax(widths)
        largest_peak = ((int(bins[peaks[i]]+bins[peaks[i]+1])//2-widths[i]//2), 
                            int((bins[peaks[i]]+bins[peaks[i]+1])//2+widths[i]//2)) # beginning and end of the peak

        if display_plots:
            ax = plt.gca()
            ax.set_xlim([0, 255])
            #Plot values in this channel
            plt.plot(bins[1:],hist, label=labels[channel], color=colors[channel])
            # Plot peaks
            plt.plot(bins[peaks+1], hist[peaks], "x")
            # Plot peak widths
            plt.hlines(hist[peaks]*0.9, bins[peaks+1]-widths//2, bins[peaks+1]+widths//2)

        return largest_peak

    if display_plots:
        fig = plt.figure(hash(title))
        plt.clf()

    background = []
    for channel in range(3):
        background.append(find_widest_peak(channel, display_plots))

    if display_plots:
        plt.title(title)
        plt.legend()
        plt.draw()

    return ([background[0][0],background[1][0],background[2][0]], 
                [background[0][1],background[1][1],background[2][1]])

def filter_out_highest_peak(frame, display_plots=False, title=None, labels=None, colors=None):
    background_thresh = find_peak_ranges(frame, display_plots, title, labels, colors)

    background_mask = cv2.bitwise_not(cv2.bitwise_or(
                        cv2.inRange(frame[:, :, 0], background_thresh[0][0], background_thresh[1][0]),
                        cv2.inRange(frame[:, :, 1], background_thresh[0][1], background_thresh[1][1]),
                        cv2.inRange(frame[:, :, 2], background_thresh[0][2], background_thresh[1][2])
                    ))
    no_background = cv2.bitwise_and(frame,frame, mask=background_mask)

    return no_background

###########################################
# Main Body
###########################################

if __name__ == "__main__":
    test_hsv_thresholds = init_test_hsv_thresholds(thresholds_used)

    while (1):
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            thresholds_used = test_hsv_thresholds(frame, thresholds_used)
            thresholds_used, thresh_frame = hsv_threshold(frame, thresholds_used)

            hsv_thresh_frame = cv2.cvtColor(thresh_frame, cv2.COLOR_BGR2HSV)

            no_bgr = filter_out_highest_peak(frame, True, "BGR peaks original", ('B','G','R'), ("blue","green","red"))
            no_hsv = filter_out_highest_peak(hsv_frame, True, "HSV peaks original", ('H','S','V'), ("red","purple","gray"))
            
            # Seeing what happens when we apply both filters consecutively
            no_bgr_hsv = filter_out_highest_peak(cv2.cvtColor(no_bgr, cv2.COLOR_BGR2HSV), True, "After BGR filtering", ('H','S','V'), ("red","purple","gray"))
            no_hsv_bgr = filter_out_highest_peak(cv2.cvtColor(no_hsv, cv2.COLOR_HSV2BGR), True, "After HSV filtering", ('B','G','R'), ("blue","green","red"))

            cv2.imshow('bgr no background', no_bgr)
            cv2.imshow('hsv no background', cv2.cvtColor(no_hsv, cv2.COLOR_HSV2BGR))
            cv2.imshow('bgr-hsv no background', cv2.cvtColor(no_bgr_hsv, cv2.COLOR_HSV2BGR))
            cv2.imshow('hsv-bgr no background', no_hsv_bgr)

            disp_hist(thresh_frame, "ideal BGR after threshold", ('B','G','R'), ("blue","green","red"))
            disp_hist(no_bgr, "actual BGR after threshold", ('B','G','R'), ("blue","green","red"))
            disp_hist(hsv_thresh_frame, "ideal HSV after threshold", ('H','S','V'), ("red","purple","gray"))
            disp_hist(no_hsv, "actual HSV after threshold", ('H','S','V'), ("red","purple","gray"))

            # update all of the plt charts
            plt.pause(0.001)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
    cv2.destroyAllWindows()
    cap.release()