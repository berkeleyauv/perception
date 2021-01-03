import argparse
import os

from perception.vis.FrameWrapper import FrameWrapper
import cv2 as cv
from perception.vis.window_builder import Visualizer
import cProfile as cp
import pstats
import imageio
from matplotlib.pyplot import Figure
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Visualizes perception algorithms.')
parser.add_argument(
    '--data', default='webcam', type=str
)
parser.add_argument('--algo_folder', default='vis.TestTasks', type=str)
parser.add_argument('--algorithm', type=str)
parser.add_argument('--cProfiler', type=str)
parser.add_argument('--save_video', action='store_true')
args = parser.parse_args()

# Get algorithm module
exec("from perception.{0}.{1} import {1} as Algorithm".format(args.algo_folder, args.algorithm))

# Initialize image source
# detects args.data, get a list of all file directory when given a directory
# change data_source to a list of all files in the directory
if os.path.isfile(args.data):
    data_sources = [args.data]
elif os.path.isdir(args.data):
    data_sources = os.listdir(args.data)
data = FrameWrapper(data_sources, 0.25)

algorithm = Algorithm()
window_builder = Visualizer(algorithm.var_info())
video_frames = []


# Main Loop
def main():
    for frame in data:

        state, debug_frames = algorithm.analyze(
            frame, debug=True, slider_vals=window_builder.update_vars()
        )

        for i, dframe in enumerate(debug_frames):
            if type(dframe) == Figure:
                img = np.fromstring(dframe.canvas.tostring_rgb(), dtype=np.uint8,
                                    sep='')
                img = img.reshape(dframe.canvas.get_width_height()[::-1] + (3,))
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                img = cv.resize(img, (frame.shape[1], frame.shape[0]))
                debug_frames[i] = img

        to_show = window_builder.display(debug_frames)
        cv.imshow('Debug Frames', to_show)
        if args.save_video:
            video_frames.append(to_show)

        key_pressed = cv.waitKey(60) & 0xFF
        if key_pressed == 112:
            cv.waitKey(0)  # pause
        if key_pressed == 113:
            break  # quit


cp.run('main()', 'algo_stats')
cv.destroyAllWindows()
p = pstats.Stats('algo_stats')

if args.cProfiler:
    p.print_stats(args.cProfiler)
else:
    p.print_stats()

if args.save_video:
    height, width, _ = video_frames[0].shape
    w = imageio.get_writer('deb_cap.mp4')
    for img in video_frames:
        height2, width2, _ = img.shape
        if (height2, width2) == (height, width):
            imag = cv.resize(img, (width - (width % 16), height - (height % 16)))
            imag = cv.cvtColor(imag, cv.COLOR_BGR2RGB)
            w.append_data(imag)
    w.close()
