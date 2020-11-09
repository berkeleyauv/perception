import argparse
from pathlib import Path
from FrameWrapper import FrameWrapper
import cv2 as cv
from window_builder import Visualizer
import cProfile as cp
import pstats

# import TestTasks.testAlgo
# Collect available datasets
"""
data_sources = ['webcam']
datasets = Path('./datasets')
for file in datasets.iterdir():
    data_sources.append(file.stem)
"""

# Parse arguments
parser = argparse.ArgumentParser(description='Visualizes perception algorithms.')
parser.add_argument(
    '--data', default='webcam', type=str
)  # do this later #, choices = data_sources)
parser.add_argument('--algorithm', type=str)
parser.add_argument('--save_video', action='store_true')
args = parser.parse_args()

# Get algorithm module
exec("from TestTasks.{} import {} as Algorithm".format(args.algorithm, args.algorithm))

# Initialize image source
data_sources = [args.data]
data = FrameWrapper(data_sources, 0.25)

# TODO: This is undefined and should be added later.
algorithm = Algorithm()
window_builder = Visualizer(algorithm.var_info())
video_frames = []
# Main Loop
def main():
    for frame in data:
        # TODO: benchmarking

        state, debug_frames = algorithm.analyze(
            frame, debug=True, slider_vals=window_builder.update_vars()
        )
        # cv.imshow('original', frame)
        to_show = window_builder.display(debug_frames)
        cv.imshow('Debug Frames', to_show)
        if args.save_video:
            video_frames.append(to_show)

        key_pressed = cv.waitKey(60) & 0xFF
        if key_pressed == 112:
            cv.waitKey(0) # pause
        if key_pressed == 113:
            break # quit


cp.run('main()', 'algo_stats')
cv.destroyAllWindows()
p = pstats.Stats('algo_stats')
p.print_stats('analyze')

if args.save_video:
    height, width, _ = video_frames[0].shape
    out = cv.VideoWriter('deb_cap.avi', cv.VideoWriter_fourcc(*'XVID'), 60, (height, width))
    for img in video_frames:
        height2, width2, _ = img.shape
        if (height2, width2) == (height, width):
            out.write(img)
    out.release()