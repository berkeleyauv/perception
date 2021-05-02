from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Tuple

from perception.tasks.segmentation.combinedFilter import CombinedFilter
import numpy as np
import cv2 as cv


class GateSegmentationAlgoA(TaskPerceiver):
    center_x_locs, center_y_locs = [], []
    
    def __init__(self):
        super().__init__()
        self.combined_filter = CombinedFilter().combined_filter

    # TODO: fix return typing
    def analyze(self, frame: np.ndarray, debug: bool=False, slider_vals=None) -> Tuple[float, float]:
        """Takes in the background removed image and returns the center between
        the two gate posts.
        Args:
            frame: The background removed frame to analyze
            debug: Whether or not tot display intermediate images for debugging
        Reurns:
            (x,y) coordinate with center of gate
        """
        rect1, rect2 = None, None

        filtered_frame = self.combined_filter(frame)[2]

        max_brightness = max([b for b in filtered_frame[:, :, 0][0]])
        lowerbound = max(0.84 * max_brightness, 120)
        upperbound = 255
        _, thresh = cv.threshold(filtered_frame, lowerbound, upperbound, cv.THRESH_BINARY)
        debug_filter = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        
        cnt = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
        
        area_diff = []
        area_cnts = []
        boundRect = None

        # remove all contours with zero area
        cnt = [cnt[i] for i in range(len(cnt)) if cv.contourArea(cnt[i]) > 0]
        for c in cnt:
            area_cnt = cv.contourArea(c)
            area_cnts.append(area_cnt)
            area_rect = cv.boundingRect(c)[-2] * cv.boundingRect(c)[-1]
            area_diff.append(abs((area_rect - area_cnt) / area_cnt))

        if len(area_diff) >= 2:
            largest_area_idx = [area_cnts.index(sorted(area_cnts, reverse=True)[i]) for i in range(min(3, len(cnt)))]
            area_diff_copy = sorted([area_diff[i] for i in largest_area_idx])
            min_i1, min_i2 = area_diff.index(area_diff_copy[0]), area_diff.index(area_diff_copy[1])

            rect1 = cv.boundingRect(cnt[min_i1])
            rect2 = cv.boundingRect(cnt[min_i2])
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            cv.rectangle(debug_filter, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), 2)
            cv.rectangle(debug_filter, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 255), 2)

            """ for csv pytest"""
            act_y = min(y1, y2)
            boundRect = (x1, act_y, np.abs(x2 - x1) + w2, max(y1 - act_y + h1, y2 - act_y + h2))
            cv.rectangle(debug_filter, (int(boundRect[0]), int(boundRect[1])), \
                         (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 255, 0), 2)

        if debug:
            return (rect1, rect2), (frame, debug_filter)
        if boundRect:
            return {'gate_box': np.array(boundRect)}
        else:
            return {'gate_box': np.zeros(4)}
        

if __name__ == '__main__':
    from perception.vis.vis import run
    run(['..\..\..\data\GOPR1142.MP4'], GateSegmentationAlgoA(), False)
