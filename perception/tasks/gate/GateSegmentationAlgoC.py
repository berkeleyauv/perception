from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Tuple
from collections import namedtuple

from perception.tasks.segmentation.combinedFilter import init_combined_filter
import numpy as np
import cv2 as cv


class GateSegmentationAlgoC(TaskPerceiver):
    __past_centers = []
    __ema = None
    output_class = namedtuple("GateOutput", ["centerx", "centery"])
    output_type = {'centerx': np.int16, 'centery': np.int16}

    def __init__(self, alpha=0.1):
        super().__init__()
        self.__alpha = alpha
        self.combined_filter = init_combined_filter()

    # TODO: fix return typing
    def analyze(self, frame: np.ndarray, debug: bool) -> Tuple[float, float]:
        """Takes in the background removed image and returns the center between
        the two gate posts.
        Args:
            frame: The background removed frame to analyze.
            debug: Whether or not to display intermediate images for debugging.
        Returns:
            (x,y) coordinate with center of gate
        """
        gate_center = self.output_class(250, 250)
        filtered_frame = self.combined_filter(frame, display_figs=False)
        filtered_frame_copies = [filtered_frame for _ in range(3)]
        stacked_filter_frames = np.concatenate(filtered_frame_copies, axis=2)
        mask = cv.inRange(
            stacked_filter_frames, np.array([100, 100, 100]), np.array([255, 255, 255])
        )
        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            contours.sort(key=self.findStraightness, reverse=True)
            cnts = contours[:2]
            rects = [cv.minAreaRect(c) for c in cnts]
            centers = [np.array(r[0]) for r in rects]
            boxpts = [cv.boxPoints(r) for r in rects]
            box = [np.int0(b) for b in boxpts]
            for b in box:
                cv.drawContours(stacked_filter_frames, [b], 0, (0, 0, 255), 5)
            if len(centers) >= 2:
                gate_center = (centers[0] + centers[1]) * 0.5
                if self.__ema is None:
                    self.__ema = gate_center
                else:
                    self.__ema = (
                        self.__alpha * gate_center + (1 - self.__alpha) * self.__ema
                    )
                gate_center = (int(self.__ema[0]), int(self.__ema[1]))
                # TODO: clean this up via hyperparam or move to gate center algo
                # if len(self.__past_centers) < 15:
                # 	self.__past_centers += [gate_center]
                # else:
                # 	self.__past_centers.pop(0)
                # 	self.__past_centers += [gate_center]
                # gate_center = sum(self.__past_centers) / len(self.__past_centers)
                # gate_center = (int(gate_center[0]), int(gate_center[1]))
                cv.circle(stacked_filter_frames, gate_center, 10, (0, 255, 0), -1)

        if debug:
            return (gate_center[0], gate_center[1]), (frame, stacked_filter_frames)
        return (gate_center[0], gate_center[1])

    def findStraightness(self, contour):  # output number = contour area/convex area, the bigger the straightest
        hull = cv.convexHull(contour, False)
        contour_area = cv.contourArea(contour)
        hull_area = cv.contourArea(hull)
        return 10 * contour_area - 5 * hull_area


if __name__ == '__main__':
    from perception.vis.vis import run
    run(['..\..\..\data\GOPR1142.MP4'], GateSegmentationAlgoC(), False)
