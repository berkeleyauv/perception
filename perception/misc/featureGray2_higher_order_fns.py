import cv2 as cv
from sys import argv as args
import numpy as np
import numpy.linalg as LA

#Jenny -> unsigned ints fixed the problem
#Damas -> flip weight vector every frame
if __name__ == "__main__":
    cap = cv.VideoCapture('../data/course_footage/path_marker_GOPR1142.mp4')
paused = False
speed = 1
#man/min of past ten frames; average or total
def init_aggregate_rescaling(only_once=False, weights=[], max_min={'max': 90, 'min': -20}): #you only pca once
    def aggregate_rescaling(frame, display_fig=False):
        nonlocal only_once, weights, max_min
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        #kernel = np.ones((5,5),np.float32)/25
        #frame = cv.filter2D(frame,-1,kernel)

        r, c, d = frame.shape
        A = np.reshape(frame, (r * c, d))

        if not only_once:

            A_dot = A - A.mean(axis=0)[np.newaxis, :]

            _, eigv = LA.eigh(A_dot.T @ A_dot)
            weights = eigv[:, 0]
            #if (weights<0).sum() > 0:
            #if np.mean(weights) < 0:
            #   weights *= -1

            red = np.reshape(A_dot @ weights, (r, c))
            only_once = True
        else:
            red = np.reshape(A @ weights, (r, c))
        #red /= np.max(np.abs(red),axis=0) #this looks real cool - Damas
        """
        if len(max_min['max']) == 10:
            max_min['max'] = max_min['max'][1:] + [np.max(red)]
            max_min['min'] = max_min['min'][1:] + [np.min(red)]
        else:
            max_min['max'].append(np.max(red))
            max_min['min'].append(np.min(red))
        """

        if np.min(red) < max_min['min']:
            max_min['min'] = np.min(red)
        if np.max(red) > max_min['max']:
            max_min['max'] = np.max(red)

        #print(np.min(red), np.max(red), 'all time Domas', max_min['min'], max_min['max'])

        red -= max_min['min']
        red *= (255.0/(max_min['max'] - max_min['min']))

        #red -= np.min(max_min['min'])
        #red *= (255.0/np.abs(np.max(max_min['max'])))

        #red -= np.min(red)
        #red *= (255.0/np.abs(np.max(red)))

        red = red.astype(np.uint8)
        red = np.expand_dims(red, axis = 2)
        red = np.concatenate((red, red, red), axis = 2)
        
        if display_fig:
            cv.imshow('One Time PCA plus all time aggregate rescaling', red)
            cv.imshow('frame', frame_gray)
        return red

    return aggregate_rescaling

if __name__ == "__main__":
    aggregate_rescaling = init_aggregate_rescaling()
    while True:
        if not paused:
            for _ in range(speed):
                ret, frame = cap.read()
        if ret:
            aggregate_rescaling(frame, True)
            #break
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('i') and speed > 1:
            speed -= 1
        if key == ord('o'):
            speed += 1