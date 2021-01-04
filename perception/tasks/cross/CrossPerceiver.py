from collections import namedtuple
import numpy as np
import sys
from perception.tasks.TaskPerceiver import TaskPerceiver

class CrossPerceiver(TaskPerceiver):
    named_tuple = namedtuple("CrossOutput", ["centerx", "centery"])
    named_tuple_types = {centerx: np.int16, centery: np.int16}
