from typing import Any
import numpy as np
class TaskPerceiver:

    def __init__(self):
        self.time = 0

    def analyze(self, frame: np.ndarray, debug: bool) -> Any:
        """Runs the algorithm and returns the result.
        Args:
			frame: The frame to analyze
            debug: Whether or not to display intermediate images for debugging

		Returns:
			the result of the algorithm
        """
        raise NotImplementedError("Need to implement with child class.")

