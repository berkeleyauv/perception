import cv2 as cv
import numpy as np
from .haze_removal import HazeRemovel
from .utils import threshold_color_array
# from combinedFilter import init_combined_filter 

# cap = cv.VideoCapture('/Users/karthikdharmarajan/Documents/URobotics/Course Footage/GOPR1146.MP4')
# combined_filter = init_combined_filter()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

def process_frame(frame):
    haze_removal_object = HazeRemovel(frame)
    dark_channel = haze_removal_object.get_dark_channel(haze_removal_object.I)
    A = haze_removal_object.get_atmosphere(dark_channel)
    t = haze_removal_object.get_transmission(dark_channel, A)
    recover_image = haze_removal_object.get_recover_image(A, t)
    return threshold_color_array(recover_image), t

# while cap.isOpened():
#     ret, img_in = cap.read()

#     if ret:
#         img_in = rescale_frame(img_in,30)
#         recovered_img, depth_map = process_frame(img_in)
#         thresholded_img_without_haze = cv.threshold(combined_filter(recovered_img), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
#         threshold_img_haze = cv.threshold(combined_filter(img_in), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
#         cv.imshow('img_in', img_in)
#         cv.imshow('recovered_img', recovered_img)
#         cv.imshow('thresholded_img_haze', threshold_img_haze)
#         cv.imshow('threshold_img_without_haze', thresholded_img_without_haze)
#         cv.imshow('depth_map', depth_map)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()
