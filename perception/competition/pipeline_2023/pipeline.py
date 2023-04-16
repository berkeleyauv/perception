from perception.tasks.TaskPerceiver import TaskPerceiver
import numpy as np
from typing import Dict
from perception.vis.SymbolClassifier import SymbolClassifier
import torch

class Pipeline2023(TaskPerceiver):

    def __init__(self, **kwargs):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.symbol_classifier = SymbolClassifier(self.device)

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        # TODO: Incorporate SAM
        # Try symbol classifier
        pass
