import cv2
import numpy as np
from sys import argv as args
import numpy.linalg as LA

import sys
sys.path.insert(0, '../background_removal')
from featureGray import init_aggregate_rescaling
from peak_removal_adaptive_thresholding import filter_out_highest_peak_multidim

if __name__ == "__main__":
    cap = cv2.VideoCapture(args[1])

def init_combined_filter():
    aggregate_rescaling = init_aggregate_rescaling(True)

    def combined_filter(frame, display_figs=False):
        nonlocal aggregate_rescaling
        #cv2.imshow('original', frame)
        pca_frame = aggregate_rescaling(frame) # this resizes the frame within its body
        #cv2.imshow("Nir's thingie", pca_frame)

        # __, other_frame = filter_out_highest_peak_multidim(
        #                             np.dstack([pca_frame[:,:,0]]), res=5)
        # cv2.imshow('norgb', other_frame[:,:])

        # Also considering RGB mostly gets rid of random stuff like the cables lying on the ground
        # Doesn't necessarily make the algorithm pick out orange things like the gate stuff more
        # The "science" behind the shift_color is a bit sketchy
        # Orange is approximately (66,170,255) in bgr
        shift_color = (66, 170, 244)
        __, other_frame = filter_out_highest_peak_multidim(
                            np.dstack([pca_frame[:,:,0], frame]), 
                            res=5, weights=[sum(shift_color)/(255)] + [c/255 for c in shift_color])
        # cv2.imshow('1x pca, orange shifted bgr', other_frame[:,:,0])

        other_frame = other_frame[:,:,:1]

        if display_figs:
            #cv2.imshow('original', frame)
            #cv2.imshow('pca thing', pca_frame)
            cv2.imshow('other filter thing', other_frame)
        return other_frame
    return combined_filter

"""
def init_aggregate_rescaling(show_frame=True):
    only_once = False
    weights = []
    max_min = {'max': 90, 'min': -20}
    def sanity_check(frame):
        nonlocal only_once
        print(only_once)
        only_once = True
        return frame


    def aggregate_rescaling(frame): #you only pca once
        nonlocal only_once
        nonlocal weights
        nonlocal max_min
        print(only_once, max_min, weights)
        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        r, c, d = frame.shape
        A = np.reshape(frame, (r * c, d))

        if not only_once:
            print('Jenny Denny"s')
            A_dot = A - A.mean(axis=0)[np.newaxis, :]

            _, eigv = LA.eigh(A_dot.T @ A_dot)
            weights = eigv[:, 0]

            red = np.reshape(A_dot @ weights, (r, c))
            only_once = True
        else:
            red = np.reshape(A @ weights, (r, c))

        if np.min(red) < max_min['min']:
            max_min['min'] = np.min(red)
        if np.max(red) > max_min['max']:
            max_min['max'] = np.max(red)

        red -= max_min['min']
        red *= (255.0/(max_min['max'] - max_min['min']))
        
        #if False:#not paused:
        #    print(np.min(red), np.max(red), max_min['min'], max_min['max'])
        
        red = red.astype(np.uint8)
        red = np.expand_dims(red, axis = 2)
        red = np.concatenate((red, red, red), axis = 2)
        
        if show_frame:
            #cv2.imshow('frame', frame_gray)
            cv2.imshow('One Time PCA plus all time aggregate rescaling', red)
        return red
    #return aggregate_rescaling
    return aggregate_rescaling
"""

if __name__ == "__main__":
    ret = True
    ret_tries = 0

    # for i in range(3000):
    #     cap.read()
    combined_filter = init_combined_filter()
    while 1 and ret_tries < 50:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            filtered_frame = combined_filter(frame, True)

            ret_tries = 0
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
        else:
            ret_tries += 1
