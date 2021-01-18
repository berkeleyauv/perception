from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.vis.TestTasks.saliency_mbd import get_saliency_mbd
from perception.vis.TestTasks.binarise import binarise_saliency_map
import numpy as np
from typing import Dict

class RBD(TaskPerceiver):

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        mbd = get_saliency_mbd(frame).astype('uint8')
        binary_sal = binarise_saliency_map(mbd,method='adaptive')
        return binary_sal, [frame, mbd, 255 * binary_sal.astype('uint8')]
