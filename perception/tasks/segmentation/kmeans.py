import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from sys import argv as args

# TODO: port to vis + TaskPerciever format or remove

########################################################################
# An attempt at an adaptive thresholding algorithm based on the frequency
# of pixel values ("peaks" if looking at a histogram of # pixels vs pixel value of a frame)
#
# *1. *** best of the three *** filter_out_highest_peak_multidim
#    pools together how "peak-like" each pixel is in all of the color channels
#    of the frame to make a final decision on what is the background
# 2. init_filter_out_highest_peak
#    gets rid of large peaks in many different color channels individually
# 3. remove_blotchy_chunks
#    places a mask over areas that have lots of edges, which in many cases
#    is equivalent to places with lots of noise
########################################################################


def k_means_segmentation(votes, frame_shape, num_groups=2, percentile=10):
    """ Attempts to use kmeans to segment the frame into num_group features
        (not including the background), denoted by a very large value in votes.
        votes is an output of the filter_out_highest_peak_multidim() function
        Output: frame_shape x num_groups 3D matrix. Get a group mapped to the
        frame by doing groups[:,:,group_num] """
    votes = np.float32(votes).flatten()

    # Make kmeans only consider the non-background pixels
    background = np.zeros(votes.shape)
    background[votes >= np.percentile(votes, percentile)] = 1
    cluster_data = votes[background == 0]
    cluster_indexes = np.array(range(len(votes)))[background == 0]

    # Do kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(cluster_data, num_groups, None, criteria, 10, flags)

    # Reconstruct the original votes array with background's label = -1
    label_arr = np.empty(votes.shape)
    label_arr[background == 1] = -1
    for i in range(num_groups):
        label_arr[cluster_indexes[labels.flatten() == i]] = i

    unique_labels, label_counts = np.unique(label_arr, return_counts=True)
    label_order = list(range(np.int0(np.amax(unique_labels)) + 2))  # something is erroring here
    if len(label_counts) < num_groups + 1:
        # add in a slot for the background if no background is found
        label_counts = np.insert(label_counts, 0, 0)

    label_order.sort(key=lambda x: label_counts[x])

    groups = np.empty((frame_shape[0], frame_shape[1], num_groups + 1))
    for i, l in enumerate(label_order):
        group = np.zeros(votes.shape)
        group[label_arr.flatten() == l - 1] = 255
        groups[:, :, i] = np.reshape(group, frame_shape[:2])

    # for i in range(len(unique_labels)):
    #     cv2.imshow(str(i) + " label", groups[:,:,i])

    return groups


###########################################
# Main Body
###########################################

if __name__ == "__main__":
    # For testing porpoises
    cap = cv2.VideoCapture(args[1])
    ret, frame = cap.read()
    out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0,
                          (int(frame.shape[1] * 0.4), int(frame.shape[0] * 0.4)))

    ret_tries = 0

    while (1 and ret_tries < 50):
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

            cv2.imshow('original', frame)
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