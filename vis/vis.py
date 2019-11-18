import argparse
from pathlib import Path

# Collect available datasets
data_sources = ['webcam']
datasets = Path('../../datasets')
for file in Folder.iterdir():
    data_sources.append(file.stem)

# Parse arguments
parser = argparse.ArgumentParser(description = 'Visualizes perception algorithms.')
parser.add_argument('--data', default = 'webcam', type=str, choices = data_sources)
parser.add_argument('--algorithm', type=str)
args = parser.parse_args()

# Initialize image source
if args.data == "webcam":
    data = None 
else:
    data = None # datasets.joinpath(args.data).iterdir()

# Main Loop
for frame in data:
    #TODO: benchmarking

    state, debug_frames = algorithm.analyze(frame, debug=True)

    cv.imshow('original', frame)
    cv.imshow('filtered_frame', debug_frames[0]) # TODO: Yuki's visualizer here

    if cv.waitKey(60) & 0xff == 27:
        break
