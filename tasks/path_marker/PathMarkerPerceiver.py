from collections import namedtuple
import numpy as np
import sys
sys.path.insert(0, '..')
from TaskPerceiver import TaskPerceiver

class PathMarkerPerceiver(TaskPerceiver):
    named_tuple = namedtuple("PathMarkerOutput", ["angle"])
    named_tuple_types = {angle: np.float64}