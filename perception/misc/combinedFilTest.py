import cv2
import numpy as np

from sys import argv as args
from aggregateRescaling import init_aggregate_rescaling
from peak_removal_adaptive_thresholding import filter_out_highest_peak_multidim

if __name__ == "__main__":
    if args[1] == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args[1])

# Returns a grayscale image
def init_combined_filter():
    aggregate_rescaling = init_aggregate_rescaling()

    def combined_filter(
        frame, custom_weights=None, display_figs=False, print_weights=False
    ):
        pca_frame = aggregate_rescaling(frame)  # this resizes the frame within its body

        __, other_frame = filter_out_highest_peak_multidim(
            np.dstack([pca_frame[:, :, 0], frame]),
            custom_weights=custom_weights,
            print_weights=print_weights,
        )

        other_frame = other_frame[:, :, :1]

        if display_figs:
            cv2.imshow('original', frame)
            cv2.imshow('Aggregate Rescaling via PCA', pca_frame)
            cv2.imshow('Peak Removal Thresholding after PCA', other_frame)
        return other_frame

    return combined_filter


if __name__ == "__main__":
    ret = True
    ret_tries = 0

    # for i in range(3000):
    #     cap.read()

    combined_filter = init_combined_filter()

    while 1 and ret_tries < 50:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
            filtered_frame = combined_filter(frame, display_figs=True)

            ret_tries = 0
            k = cv2.waitKey(60) & 0xFF
            if k == 27:
                break
        else:
            ret_tries += 1
