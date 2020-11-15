import cv2 as cv
import numpy as np
from .haze_removal import HazeRemoval
from .utils import threshold_color_array
from ...vis.TaskPerceiver import TaskPerceiver

class DepthMap(TaskPerceiver):

    def __init__(self, beta=1):
        self.beta = beta

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        haze_removal_object = HazeRemoval(frame)
        dark_channel = haze_removal_object.get_dark_channel(haze_removal_object.I)
        A = haze_removal_object.get_atmosphere(dark_channel)
        t = haze_removal_object.get_transmission(dark_channel, A)
        return np.log(t) / -self.beta
