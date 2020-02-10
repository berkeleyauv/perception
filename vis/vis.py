import argparse
from pathlib import Path
from FrameWrapper import FrameWrapper
import cv2 as cv
import sys
import importlib.util
import importlib
import yukiVisualizer
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
# Main Loop
for frame in data:
    #TODO: benchmarking

    state, debug_frames = algorithm.analyze(frame, debug=True)

    cv.imshow('original', frame)

    yukiVisualizer.display(debug_frames[:2])
    if cv.waitKey(60) & 0xff == 27:
        break
