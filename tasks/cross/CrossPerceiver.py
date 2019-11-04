from collections import namedtuple
import numpy as np
import sys
sys.path.insert(0, '..')
from TaskPerceiver import TaskPerceiver

class CrossPerceiver(TaskPerceiver):
    named_tuple = namedtuple("CrossOutput", ["centerx", "centery"])
    named_tuple_types = {centerx: np.int16, centery: np.int16}