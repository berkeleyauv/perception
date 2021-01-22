import cv2
import numpy as np

from perception.tasks.segmentation.aggregateRescaling import init_aggregate_rescaling
from perception.tasks.segmentation.peak_removal_adaptive_thresholding import filter_out_highest_peak_multidim
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict, Tuple

class CombinedFilter(TaskPerceiver):

    def __init__(self):
        super().__init__()
        self.aggregate_rescaling = init_aggregate_rescaling(False)

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]=None) -> Tuple[float, float]:
        filtered_frames = self.combined_filter(frame)

        if debug:
            return None, filtered_frames # returns None because it's a more general algorithm
        return None

    def combined_filter(self, frame, custom_weights=None, print_weights=False):
        pca_frame = self.aggregate_rescaling(frame) # this resizes the frame within its body

        __, other_frame = filter_out_highest_peak_multidim(
                            np.dstack([pca_frame[:,:,0], frame]),
                            custom_weights=custom_weights,
                            print_weights=print_weights)

        other_frame = other_frame[:, :, :1]

        return [frame, pca_frame, other_frame]

if __name__ == "__main__":
    from perception.vis.vis import run

    run(['..\..\..\data\GOPR1142.MP4'], CombinedFilter(), False)
