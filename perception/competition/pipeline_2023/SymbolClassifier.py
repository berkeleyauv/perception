from perception.tasks.TaskPerceiver import TaskPerceiver
from typing import Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

class SymbolClassifier(TaskPerceiver):
    def __init__(self, device):
        self.model = torch.load("model.pt")
        self.model.to(device)
        self.model.eval()
    def analyze(self, frame: np.ndarray, debug: bool):
        frame_torch = torch.from_numpy(frame)
        with torch.no_grad():
             foundClass = self.model(frame_torch)
        if debug:
            cv2.putText(frame, str(foundClass), (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 20, (255, 0, 0), thickness=5)
            return foundClass, [frame]
        return foundClass, []

if __name__ == '__main__':
    from perception.vis.vis import run
    run(['webcam'], SymbolClassifier(), True)

