from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.segmentation.saliency_detection.saliency_mbd import get_saliency_mbd
import numpy as np
import cv2 as cv
from typing import Dict

class MBD(TaskPerceiver):

    def __init__(self, **kwargs):
        super().__init__(lower_bound=((0,255), 10))

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        mbd = get_saliency_mbd(frame).astype('uint8')
        return mbd, [frame, mbd]
