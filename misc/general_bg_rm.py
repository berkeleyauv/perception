import cv2 as cv
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("file", help="file")
args = parser.parse_args()
from dark_channel.handler import process_frame as dark_channel



def DarkChannel(im):
    b,g,r = cv.split(im)
    dc = cv.min(cv.min(r,g),b);
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(im.shape[0],im.shape[1]))
    dark = cv.erode(dc,kernel)
    return dark

def resize_frame(frame,ratio = 0.4):
    return cv.resize(frame,(int(frame.shape[1]*ratio),int(frame.shape[0]*ratio)))

def save_frames(frames,folder):
    os.mkdir(folder)
    [cv.imwrite(f'{folder}/{frame}.png', frames[frame]) for frame in frames]

def show_frames(frames):
    [cv.imshow(frame,frames[frame]) for frame in frames]

def analyze(src):

    # src = cv.imread(fn);
    # src = resize_frame(src)
    img = src
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # # sure background area
    # sure_bg = cv.dilate(opening,kernel,iterations=3)
    # # Finding sure foreground area
    # dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    # ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv.subtract(sure_bg,sure_fg)

    # # Marker labelling
    # ret, markers = cv.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0
    # markers = cv.watershed(img,markers)
    # orig = resize_frame(cv.imread(fn))
    # orig[markers == -1] = [255,0,0]
    return thresh
    # frames = {'test':thresh,'dist':dist_transform,'sure_fg':sure_fg,'sure_bg':sure_bg,'unknown':unknown,'marked':orig}
    # show_frames(frames)
    # save_frames(frames,'binary_inv_dc/orig')
    # cv.waitKey()
    

if __name__ == '__main__':
    print(args.file)
    cap = cv.VideoCapture(args.file)
    while not cap.isOpened():   
        cap = cv.VideoCapture(args.file)
        cv.waitKey(1000)
        print ("Wait for the header")
    while(True):
        ret, frame = cap.read()
        if ret:
            frame = resize_frame(frame, 0.25)
            dark = dark_channel(frame)[0]
            # cv.imshow('OTSU+Dark',analyze(dark))
            show_frames({'OTSU':analyze(frame),'dark':dark,'OTSU+DARK':analyze(dark),'orig':frame})
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            if cv.waitKey(32) == ord(' '):
                while(not cv.waitKey(32) == ord(' ')):
                    continue
    cv.waitKey(-1)