from GateSegmentationAlgo2 import GateSegmentationAlgo
from GatePerceiver import GatePerceiver
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import math
import cv2 as cv
import time
import cProfile
import statistics

class GateCenter(GatePerceiver):
    def __init__(self):
        super()
        self.gate_center = self.output_class(250, 250)
    
    
    def analyze(self, frame):
        