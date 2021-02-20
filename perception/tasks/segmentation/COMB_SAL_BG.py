import cv2 as cv
import numpy as np
from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
from perception.tasks.segmentation.saliency_detection.MBD import MBD
from perception.vis.TestTasks.BackgroundRemoval import BackgroundRemoval
from threading import Thread
import queue
import concurrent.futures
from multiprocessing import Pool
from shapely.geometry import Polygon


class COMB_SAL_BG(TaskPerceiver):
    def __init__(self, **kwargs):
        super().__init__(blur = ((0, 10), 2), lamda = ((0,10),1), lower_bound=((0,255), 10))
        self.sal = MBD()
        self.bg = BackgroundRemoval()
        # self.q = queue.Queue()

    # def f(frame: np.ndarray, debug: bool, slider_vals: Dict[str, int], use_mbd: bool):
    #     print('working!')
    #     if use_mbd:
    #         print('Im HERE!!!!')
    #         ret = self.sal.analyze(frame, debug, slider_vals=slider_vals)
    #     else:
    #         print('Im NOT :( HERE!!!!')
    #         ret = self.bg.analyze(frame,debug, slider_vals=slider_vals)
    #     return ret


    # def filter_contours(self, contour_main, contour_filter):
    #     TODO: find algorithm to filter contour_main based upon contour_filter

    def intersection_over_union(polyA, polyB):
        polygonA_shape = Polygon(polyA)
        polygonB_shape = Polygon(polyB)

        polygon_intersection = polygonA_shape.intersection(polygonB_shape).area
        polygon_union = polygonA_shape.area + polygonB_shape.area - polygon_intersection #inclusion exclusion

        IOU = polygon_intersection / polygon_union
        return IOU

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        # sal_thread = Thread(target=self.sal.analyze, args=(frame, debug, slider_vals, self.q))
        # bg_thread = Thread(target=self.bg.analyze, args=(frame, debug, slider_vals, self.q))
        # bg_thread.__setattr__('_args', (frame, debug, slider_vals, self.q))
        # sal_thread.start()
        # bg_thread.start()
        # sal_thread.join()
        # bg_thread.join()
        # ret = cv.bitwise_and(self.q.get(),self.q.get())

        sal = self.sal.analyze(frame, debug, slider_vals=slider_vals)
        bg = self.bg.analyze(frame,debug, slider_vals=slider_vals)
        ret = cv.bitwise_and(sal[0],bg[0])
        return frame, [frame, sal[0], bg[0], ret]