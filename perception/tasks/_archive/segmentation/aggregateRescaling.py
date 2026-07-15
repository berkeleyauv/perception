import cv2 as cv
from sys import argv as args
import numpy as np
import numpy.linalg as LA


# TODO: port to vis + TaskPerciever format or remove
# Jenny -> unsigned ints fixed the problem
# Damas -> flip weight vector every frame

# man/min of past ten frames; average or total
def init_aggregate_rescaling(show_frame=True):
    only_once = False
    weights = []
    max_min = {'max': 90, 'min': -20}

    def aggregate_rescaling(frame):  # you only pca once
        nonlocal only_once
        nonlocal weights
        nonlocal max_min

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        r, c, d = frame.shape
        A = np.reshape(frame, (r * c, d))

        if not only_once:

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
        red *= 255.0 / (max_min['max'] - max_min['min'])
        """
        if False:#not paused:
            print(np.min(red), np.max(red), max_min['min'], max_min['max'])
        """
        red = red.astype(np.uint8)
        red = np.expand_dims(red, axis=2)
        red = np.concatenate((red, red, red), axis=2)

        if show_frame:
            cv.imshow('frame', frame_gray)
            cv.imshow('One Time PCA plus all time aggregate rescaling', red)
        return red

    return aggregate_rescaling


paused = False
speed = 1
if __name__ == '__main__':
    cap = cv.VideoCapture(args[1])
    agg_res = init_aggregate_rescaling()
    while True:
        if not paused:
            for _ in range(speed):
                ret, frame = cap.read()
        if ret:
            frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            agg_res(frame)
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('i') and speed > 1:
            speed -= 1
        if key == ord('o'):
            speed += 1
