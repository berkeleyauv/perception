import argparse
from pathlib import Path
from FrameWrapper import FrameWrapper
import cv2 as cv
import sys
import importlib.util
import importlib
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
"""
spec = importlib.util.spec_from_file_location("module.name", "TestTasks/{}".format(args.algorithm))
algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algorithm)
"""
exec("from TestTasks.{} import {} as Algorithm".format(args.algorithm, args.algorithm))
# Initialize image source
data = FrameWrapper(data_sources, .5)

algorithm = Algorithm()
# Main Loop
for frame in data:
    #TODO: benchmarking

    state, debug_frames = algorithm.analyze(frame, debug=True)

    cv.imshow('original', frame)
    cv.imshow('filtered_frame', debug_frames[0]) # TODO: Yuki's visualizer here

    if cv.waitKey(60) & 0xff == 27:
        break
