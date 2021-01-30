from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.segmentation.saliency_detection.saliency_mbd import get_saliency_mbd
import numpy as np
import cv2 as cv
from typing import Dict

class MBD(TaskPerceiver):

    def __init__(self, **kwargs):
        super().__init__(lower_bound=((0,255), 10))

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        width = int(frame.shape[1] * 0.7)
        height = int(frame.shape[0] * 0.7)
        dsize = (width, height)
        frame = cv.resize(frame, dsize)
        mbd = get_saliency_mbd(frame).astype('uint8')
        ret3,th3 = cv.threshold(mbd,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return th3, [frame, mbd, th3]
