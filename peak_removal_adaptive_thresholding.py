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

# Notes on performance
# - Does worse when the target object is really big, especially if
#    it is one uniform color
# - Lets through objects of multiple colors at once because only 
#    "background" stuff is touched
# - Will prioritize large weirdly-colored objects, like the flipper
#    in the play slots video

# TODO: do low pass filter and close kernel
# TODO: https://stackoverflow.com/questions/37221685/filtering-foliage-clutter-in-an-image-opencv
#       for getting rid of edgy areas <lenny>
# TODO: maybe change the number of times it filters in hsv based on the colors in the task
#           ex: hsv bgr hsv takes care of the background. If there's a big white object, hsv again?

peak_width_height = 0.95 # How far down the peak that the algorithm draws 
                        # the horizontal width line
lpf_lambda = 0.9
lpf_cache_size = 10

testing = False

peak_filters = ['hsv', 'bgr', 'hsv']

# cap = cv2.VideoCapture('../data/course_footage/GOPR1142.MP4')
# # No thresholds
# h_low = 0
# s_low = 0
# v_low = 0
# h_hi = 255
# s_hi = 255
# v_hi = 255

cap = cv2.VideoCapture('../data/course_footage/path_marker_GOPR1142.mp4')
# Path marker default
h_low = 31
s_low = 28
v_low = 179
h_hi = 79
s_hi = 88
v_hi = 218

# cap = cv2.VideoCapture('../data/course_footage/play_slots_GOPR1142.MP4')
# # Play slots default
# h_low = 96
# s_low = 82
# v_low = 131
# h_hi = 190
# s_hi = 180
# v_hi = 228

thresholds_used = [h_low, s_low, v_low, h_hi, s_hi, v_hi]

def init_test_hsv_thresholds(thresholds):
    # Keep track of previous threhold values to see if the user is using the trackbar
    # is there a function that detects whether the mouse button is down?
    prev_h_low, prev_s_low, prev_v_low, prev_h_hi, prev_s_hi, prev_v_hi = thresholds

    def nothing(x):
        """Helper method for the trackbar"""
        pass

    cv2.namedWindow('ideal thresholding')
    cv2.createTrackbar('h_low','ideal thresholding',h_low,255,nothing)
    cv2.createTrackbar('s_low','ideal thresholding',s_low,255,nothing)
    cv2.createTrackbar('v_low','ideal thresholding',v_low,255,nothing)
    cv2.createTrackbar('h_high','ideal thresholding',h_hi,255,nothing)
    cv2.createTrackbar('s_high','ideal thresholding',s_hi,255,nothing)
    cv2.createTrackbar('v_high','ideal thresholding',v_hi,255,nothing)

    def test_hsv_thresholds(frame, thresholds):
        nonlocal prev_h_low, prev_s_low, prev_v_low, prev_h_hi, prev_s_hi, prev_v_hi

        h_low_track = cv2.getTrackbarPos('h_low','ideal thresholding')
        s_low_track = cv2.getTrackbarPos('s_low','ideal thresholding')
        v_low_track = cv2.getTrackbarPos('v_low','ideal thresholding')
        h_hi_track = cv2.getTrackbarPos('h_high','ideal thresholding')
        s_hi_track = cv2.getTrackbarPos('s_high','ideal thresholding')
        v_hi_track = cv2.getTrackbarPos('v_high','ideal thresholding')

        if h_low_track!=prev_h_low or s_low_track!=prev_s_low or v_low_track!=prev_v_low \
                or h_hi_track!=prev_h_hi or s_hi_track!=prev_s_hi or v_hi_track!=prev_v_hi:
            # If user is adjusting the trackbars, use the user input
            thresholds_used = [h_low_track, s_low_track, v_low_track, h_hi_track, s_hi_track, v_hi_track]
        else:
            # Otherwise, copy program data to trackbars
            thresholds_used = thresholds
            cv2.setTrackbarPos('h_low', 'ideal thresholding', thresholds_used[0])
            cv2.setTrackbarPos('s_low', 'ideal thresholding', thresholds_used[1])
            cv2.setTrackbarPos('v_low', 'ideal thresholding', thresholds_used[2])
            cv2.setTrackbarPos('h_high', 'ideal thresholding', thresholds_used[3])
            cv2.setTrackbarPos('s_high', 'ideal thresholding', thresholds_used[4])
            cv2.setTrackbarPos('v_high', 'ideal thresholding', thresholds_used[5])

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(thresholds_used[:3]), np.array(thresholds_used[3:]))
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('ideal thresholding', res)

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
        Result is formatted to fit cv2.inRange() -> ((low1, low2, low3), (hi1, hi2, hi3))

        Shape of frame must have 3 dimensions (pass in np.expand_dims(frame, 2) if erroring) """

    # TODO: Maybe use a different combination of peak characteristics to more accurately
    #       select the entire peak (only the tip is selected right now)

    def find_highest_peak(channel, display_plots=False):
        """ Finds and returns the x-range of the highest peak in the
            given channel of frame """

        f = frame[:, :, channel].flatten()

        # Some semi-hardcoded values :)
        num_bins = max(int((np.amax(f)-np.amin(f)) / 4), 10)

        hist, bins = np.histogram(f, bins=num_bins)

        hist[0] = 0 # get rid of stuff that was thresholded to 0
        hist = np.hstack([hist, [0]]) # make stuff at 255 into a peak
        bins = np.hstack([bins, [bins[bins.shape[0]-1] + 1]])

        peaks, properties = find_peaks(hist, height=0.1)
        if len(peaks) > 0:
            i = np.argmax(properties['peak_heights'])
            widths = peak_widths(hist, peaks, rel_height=peak_width_height)[0]
            # i = np.argmax(widths)
            largest_peak = (int((bins[peaks[i]]+bins[peaks[i]+1])//2-widths[i]//2), 
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
        else:
            largest_peak = (0, 0)

        return largest_peak

    if display_plots:
        fig = plt.figure(hash(title))
        plt.clf()

    background = (np.empty(frame.shape[2]),np.empty(frame.shape[2]))
    for channel in range(frame.shape[2]):
        low, high = find_highest_peak(channel, display_plots)
        background[0][channel] = low
        background[1][channel] = high

    if display_plots:
        plt.title(title)
        plt.legend()
        plt.draw()

    return background

def plot_peaks(frame, title, labels, colors):
    # Shh this is just a helper function that makes the code more readable
    # Not to be used in practice.
    find_peak_ranges(frame, True, title, labels, colors)

def init_filter_out_highest_peak(filters, return_colorspace="any", input_colorspace="bgr"):
    """ Takes in an hsv image! Returns an hsv image"""
    # low pass filter
    # vk* = vk*lambda + v*(1-lambda)
    # lambda = 0.9-0.4

    prev_hsv_threshes = [[] for i in range(len(filters))]
    hsv_labels = (('H','S','V'), ("red","purple","gray"))
    bgr_labels = (('B','G','R'), ("blue","green","red"))

    # Figure out how the procedure to convert among hsv and bgr.
    # Format of stuff in fitler_fns:
    # [<'c' convert or 'f' filter>, <target colorspace>]
    filter_fns = []
    curr_color = input_colorspace
    for f in filters:
        if f != curr_color:
            filter_fns.append(['c',f])
            curr_color = f
        filter_fns.append(['f',f])
    if return_colorspace != "any" and return_colorspace != curr_color:
        filter_fns.append(['c', return_colorspace])

    def filter_out_highest_peak(frame, cache, display_plots=False, title=None, labels=None, colors=None):

        background_thresh = find_peak_ranges(frame, display_plots, title, labels, colors)
        raw_thresh = background_thresh
        # multiply everything in cache by (1-lpf_lambda)
        if len(cache) > 0:
            # cache = np.array(cache) * (1-lpf_lambda)
            # calculate average
            for i in range(2):
                for j in range(3):
                    background_thresh[i][j] = (background_thresh[i][j] + sum([c[i][j] for c in cache])) // (len(cache) + 1)

        background_mask = cv2.bitwise_not(cv2.bitwise_or(
                            cv2.inRange(frame[:, :, 0], background_thresh[0][0], background_thresh[1][0]),
                            cv2.inRange(frame[:, :, 1], background_thresh[0][1], background_thresh[1][1]),
                            cv2.inRange(frame[:, :, 2], background_thresh[0][2], background_thresh[1][2])
                        ))
        no_background = cv2.bitwise_and(frame,frame, mask=background_mask)

        return background_thresh, raw_thresh, no_background

    def combine_threshes(th1, th2):
        return ([min(th1[0][0], th2[0][0]), min(th1[0][1], th2[0][1]), min(th1[0][2], th2[0][2])], 
                    [max(th1[1][0], th2[1][0]), max(th1[1][1], th2[1][1]), max(th1[1][2], th2[1][2])])

    def bgr_thresh2hsv_thresh(th):
        th = cv2.cvtColor(np.array([[th[0]], [th[1]]], np.uint8), cv2.COLOR_BGR2HSV).tolist()
        return ([min(th[0][0][0], th[1][0][0]), min(th[0][0][1], th[1][0][1]), min(th[0][0][2], th[1][0][2])],
                [max(th[0][0][0], th[1][0][0]), max(th[0][0][1], th[1][0][1]), max(th[0][0][2], th[1][0][2])])


    def do_filter(frame, display_plots=False):
        nonlocal prev_hsv_threshes
        if len(prev_hsv_threshes[0]) == lpf_cache_size:
            for x in prev_hsv_threshes:
                x.pop(0)

        filter_index = 0
        for f in filter_fns:
            if f[0] == 'c':
                if f[1] == "hsv":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                else:
                    # f[1] == "bgr"
                    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            else:
                if f[1] == "hsv":
                    thresh, raw_thresh, frame = filter_out_highest_peak(frame, prev_hsv_threshes[filter_index])
                    prev_hsv_threshes[filter_index].append(raw_thresh)
                else:
                    # f[1] == "bgr"
                    thresh, raw_thresh, frame = filter_out_highest_peak(frame, prev_hsv_threshes[filter_index])
                    thresh = bgr_thresh2hsv_thresh(thresh)
                    prev_hsv_threshes[filter_index].append(raw_thresh)
                filter_index += 1

        # Doesn't do anything :c
        # frame = cv2.fastNlMeansDenoising(frame)

        # # # Post processing
        # # Performs badly if there is a lot of noise or if there is no noise at all around targets
        # frame = remove_blotchy_chunks(frame, iterations=1, display_imgs=True)
        # cv2.imshow('after antiblotchy', frame)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('after open', frame)

        return frame

    return do_filter

def keep_highest_valued_peaks_mask(frame, num_peaks=1, display_plots=False, title=None, label='1', color='blue'):
    """ Returns a mask for the frame that keeps the num_peaks highest peaks in the histogram of
        pixel values.
        Only works for grayscale/1-channel images (to speed this up) 
        Shape of frame must have 3 dimensions (pass in np.expand_dims(frame, 2) if erroring) """
    # Some semi-thresholded values :)
    num_bins = max(int((np.amax(frame)-np.amin(frame)) / 4), 10)
    hist, bins = np.histogram(frame, bins=num_bins)
    hist[0] = 0 # get rid of stuff that was thresholded to 0
    hist = np.hstack([hist, [0]]) # make stuff at 255 into a peak
    bins = np.hstack([bins, [bins[bins.shape[0]-1] + 1]])

    peaks, properties = find_peaks(hist, prominence=100)
    widths = peak_widths(hist, peaks, rel_height=peak_width_height)[0]

    if len(peaks) > 0:
        i = len(peaks) - 1
        mask = cv2.inRange(frame, (bins[peaks[i]]+bins[peaks[i]+1])//2-widths[i]*2,
                                    (bins[peaks[i]]+bins[peaks[i]+1])//2+widths[i]*2)

        # To support keeping multiple peaks
        for j in range(num_peaks - 1):
            i = len(peaks) - 2 - j
            if i >= 0:
                mask = cv2.bitwise_or(cv2.inRange(frame, (bins[peaks[i]]+bins[peaks[i]+1])//2-widths[i],
                                        (bins[peaks[i]]+bins[peaks[i]+1])//2+widths[i]), mask)
        # frame = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        mask = np.ones(frame.shape, np.uint8)

    if display_plots:
        fig = plt.figure(hash(title))
        plt.clf()

        ax = plt.gca()
        ax.set_xlim([0, 255])
        #Plot values in this channel
        plt.plot(bins[1:],hist, label=label, color=color)
        # Plot peaks
        plt.plot(bins[peaks+1], hist[peaks], "x")
        # Plot peak widths
        plt.hlines(hist[peaks]*0.9, bins[peaks+1]-widths//2, bins[peaks+1]+widths//2)

        plt.title(title)
        plt.legend()
        plt.draw()

    return mask

def delete_lowest_valued_peaks_mask(frame, num_peaks=1, display_plots=False, title=None, label='1', color='blue'):
    """ Returns a mask for the frame that deletes the num_peaks lowest-valued peaks in the histogram of
        pixel values.
        Only works for grayscale/1-channel images (to speed this up) """

    # Some semi-thresholded values :)
    num_bins = max(int((np.amax(frame)-np.amin(frame)) / 4), 10)
    hist, bins = np.histogram(frame, bins=num_bins)
    hist[0] = 0 # get rid of stuff that was thresholded to 0

    peaks, properties = find_peaks(hist, prominence=100)
    widths = peak_widths(hist, peaks, rel_height=peak_width_height)[0]

    if len(peaks) > 0:
        i = 0
        mask = cv2.bitwise_not(cv2.inRange(frame, (bins[peaks[i]]+bins[peaks[i]+1])//2-widths[i]*2,
                                    (bins[peaks[i]]+bins[peaks[i]+1])//2+widths[i]*2))

        # To support deleting multiple peaks
        for j in range(num_peaks - 1):
            i = j + 1
            if len(peaks) > i:
                mask = cv2.bitwise_and(cv2.bitwise_not(cv2.inRange(
                            frame, (bins[peaks[i]]+bins[peaks[i]+1])//2-widths[i]*2,
                                (bins[peaks[i]]+bins[peaks[i]+1])//2+widths[i]*2)), mask)
    else:
        mask = np.ones(frame.shape, np.uint8)
        # frame = cv2.bitwise_and(frame, frame, mask=mask)

    if display_plots:
        fig = plt.figure(hash(title))
        plt.clf()

        ax = plt.gca()
        ax.set_xlim([0, 255])
        #Plot values in this channel
        plt.plot(bins[1:],hist, label=label, color=color)
        # Plot peaks
        plt.plot(bins[peaks+1], hist[peaks], "x")
        # Plot peak widths
        plt.hlines(hist[peaks]*0.9, bins[peaks+1]-widths//2, bins[peaks+1]+widths//2)

        plt.title(title)
        plt.legend()
        plt.draw()

    return mask

def remove_blotchy_chunks(frame, kernel_size=201, iterations=1, display_imgs=False):
    """ Works best when object isn't surrounded by blotchy stuff """
    edges = cv2.Canny(frame, 100, 150)

    blurred = edges.copy()
    for _ in range(iterations):
        blurred = cv2.GaussianBlur(edges, (kernel_size, kernel_size), -1)

    ret, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    if display_imgs:
        cv2.imshow('original', frame)
        cv2.imshow('edges', edges)
        cv2.imshow('blurred', blurred)
        cv2.imshow('mask', mask)
        cv2.imshow('blotchy result', result)

    return result

# def single_out_convex_contour(frame):

###########################################
# Main Body
###########################################

if __name__ == "__main__":
    # For testing porpoises
    ret, frame = cap.read()
    out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (int(frame.shape[1]*0.4), int(frame.shape[0]*0.4)))

    if testing:
        test_hsv_thresholds = init_test_hsv_thresholds(thresholds_used)

    filter_peaks = init_filter_out_highest_peak(peak_filters, 'hsv')

    ret_tries = 0
    while (1 and ret_tries < 50):
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

            filtered = filter_peaks(frame)

            # some more things to delete peaks channel by channel
            h_channel = filtered[:, :, 0]
            s_channel = filtered[:, :, 1]
            v_channel = filtered[:, :, 2]
            s_mask = delete_lowest_valued_peaks_mask(s_channel, 1)
            v_mask = delete_lowest_valued_peaks_mask(v_channel, 1)
            mask = cv2.bitwise_and(s_mask, v_mask)
            filtered2 = cv2.bitwise_and(filtered, filtered, mask=mask)

            cv2.imshow('original', frame)
            cv2.imshow('filter ' + "".join([f + " " for f in peak_filters]), filtered)
            cv2.imshow('2nd filter', filtered2)
            cv2.imshow('s_mask', s_mask)
            cv2.imshow('v_mask', v_mask)

            # For testing porpoises 
            out.write(filtered2)

            # Update all of the plt charts
            plt.pause(0.001)

            ret_tries = 0
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
        else:
            ret_tries += 1
    cv2.destroyAllWindows()
    cap.release()
    out.release()
