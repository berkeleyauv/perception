from collections import namedtuple
import numpy as np
import sys
sys.path.insert(0, '..')
from perception.tasks.TaskPerceiver import TaskPerceiver

class GatePerceiver(TaskPerceiver):
    output_class = namedtuple("GateOutput", ["centerx", "centery"])
    output_type = {'centerx': np.int16, 'centery': np.int16}
