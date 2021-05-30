import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.misc.dark_channel.handler import process_frame as dark_channel


class BackgroundRemoval(TaskPerceiver):

    def __init__(self, **kwargs):
        super().__init__(blur = ((0, 10), 2), lamda = ((0,10),1))
        self.prev_frame = None
        self.knn = cv.createBackgroundSubtractorKNN()
    
    def gen_lamda_frame(self, frame, lamda):
        return (frame*lamda).astype(np.uint8)
    def calculate_blur(self, frame, lamda):
        shape = frame.shape
        dt = frame.dtype
        if self.prev_frame is None:
            # print(frame.dtype,self.gen_lamda_frame(shape,lamda,dt).dtype)
            self.prev_frame = self.gen_lamda_frame(frame,lamda)
        else:
            self.prev_frame = cv.add(self.gen_lamda_frame(self.prev_frame,1-lamda), self.gen_lamda_frame(frame,lamda))
        return self.prev_frame

    def analyze_old(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        # print(frame.shape)
        blur = frame
        # self.calculate_blur(frame,slider_vals["lamda"]/10)
        other = dark_channel(frame)[0]
        other_dark = self.calculate_blur(other,slider_vals["blur"]/10)
        other_otsu = self.threshold(other_dark)
        dark = dark_channel(blur)[0]
        otsu_dark = self.threshold(dark)
        clh =  self.threshold(dark, clh = True)
        no_blur = self.threshold(dark_channel(frame)[0], clh = True)
        # temp = np.zeros(no_blur.shape)
        # print(no_blur.shape)
        # cv.fastNlMeansDenoising(no_blur, temp, 10,10, 7, 21)
        # print(blur.shape, dark.shape, otsu_dark.shape, clh.shape, no_blur.shape)
        return otsu_dark, [otsu_dark, clh, no_blur, blur, dark, frame, other, other_dark, other_otsu]
    
    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        knn = self.knn.apply(frame)
        return knn, [frame, knn]

    def threshold(self, frame, x = 0, y =255, clh = False):
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        if clh:
            clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(10,10))
            thresh = clahe.apply(gray)
            _, thresh = cv.threshold(thresh,x,y,cv.THRESH_BINARY+cv.THRESH_OTSU)
        else:    
            _, thresh = cv.threshold(gray,x,y,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
        return thresh

    def blur_frame(self, frame, size = 15):
        size = max(size, 1)
        return cv.blur(frame,(size,size))