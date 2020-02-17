import argparse
from pathlib import Path
from FrameWrapper import FrameWrapper
import cv2 as cv
import sys
import importlib.util
import importlib
from yukiVisualizer import Visualizer
#import TestTasks.testAlgo
# Collect available datasets
data_sources = ['webcam']
datasets = Path('./datasets')
for file in datasets.iterdir():
    data_sources.append(file.stem)

# Parse arguments
parser = argparse.ArgumentParser(description = 'Visualizes perception algorithms.')
parser.add_argument('--data', default = 'webcam', type=str, choices = data_sources)
parser.add_argument('--algorithm', type=str)
args = parser.parse_args()

# Get algorithm module
exec("from TestTasks.{} import {} as Algorithm".format(args.algorithm, args.algorithm))

# Initialize image source
data_sources = ['./datasets/{}.mp4'.format(args.data)]
data = FrameWrapper(data_sources, .25)

algorithm = Algorithm()
yukiVisualizer = Visualizer(algorithm.var_info())
# Main Loop
for frame in data:
    #TODO: benchmarking

    state, debug_frames = algorithm.analyze(frame, debug=True, slider_vals=yukiVisualizer.update_vars())
    # cv.imshow('original', frame)
    yukiVisualizer.display(debug_frames)

    if cv.waitKey(60) & 0xff == 113:
        break
