from typing import Any, Dict, Tuple
import numpy as np

class TaskPerceiver:

    def __init__(self, **kwargs):
        """Initializes the TaskPerceiver.
        Args:
            kwargs: Each keyworded argument is of the form
                var_name = (range, default_val), where range is the range of values
                for the slider which controls this variable, and default_val is
                the initial value of the slider.
        """
        self.time = 0
        self.variables = kwargs

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]) -> Any:
        """Runs the algorithm and returns the result.
        Args:
			frame: The frame to analyze
            debug: Whether or not to display intermediate images for debugging
            slider_vals: A list of names of the variables which the user should be
                able to control from the Visualizer, mapped to current slider
                value for that variable

		Returns:
			the result of the algorithm
        """
        raise NotImplementedError("Need to implement with child class.")

    def var_info(self) -> Dict[str, Tuple[Tuple[int, int], int]]:
        return self.variables
