import cv2 as cv
import numpy as np
from TaskPerceiver import TaskPerceiver
from typing import Dict
from .dark_channel.handler import process_frame as dark_channel

class BackgroundRemoval(TaskPerceiver):

    def __init__(self, **kwargs):
        super().__init__(blur = ((0, 10), 2))


    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        blur = self.blur_frame(frame, size = slider_vals['blur'])
        dark = dark_channel(blur)[0]
        otsu_dark = self.threshold(dark)
        clh =  self.threshold(dark, clh = True)
        no_blur = self.threshold(dark_channel(frame)[0], clh = True)
        # print(blur.shape, dark.shape, otsu_dark.shape, clh.shape, no_blur.shape)
        return no_blur, [otsu_dark, clh, no_blur, blur, dark, frame]

    def threshold(self, frame, x = 0, y =255, clh = False):
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        if clh:
            clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(10,10))
            thresh = clahe.apply(gray)
            _, thresh = cv.threshold(thresh,x,y,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        else:    
            _, thresh = cv.threshold(gray,x,y,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
        return thresh

    def blur_frame(self, frame, size = 15):
        size = max(size, 1)
        return cv.blur(frame,(size,size))