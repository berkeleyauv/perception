from TaskPerceiver import TaskPerceiver
from typing import Tuple
from sys import argv as args
from combinedFilter import init_combined_filter
#from segmentation.aggregateRescaling import init_aggregate_rescaling

class GateTask(TaskPerceiver):
	def analyze(self, frame: np.ndarray, debug: bool) -> Tuple[float, float]:
		"""Takes in the background removed image and returns the center between
		the two gate posts.
		Args:
			frame: The background removed frame to analyze
			debug: Whether or not tot display intermediate images for debugging

		Reurns:
			(x,y) coordinate with center of gate
		"""
		# FILL IN YOUR FILTER HERE
		return (250, 250)

#if __name__ == '__main__':
#	cap = cv.VideoCapture(args[1])
	