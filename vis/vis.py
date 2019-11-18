import argparse
from pathlib import Path

data_sources = ['webcam']
Folder = Path('../../datasets')
for file in Folder.iterdir():
    data_sources.append(file.stem)

parser = argparse.ArgumentParser(description = 'Visualizes perception algorithms.')
parser.add_argument('--data', default = 'webcam', type=str, choices = data_sources)
parser.add_argument('--algorithm', type=str)
args = parser.parse_args()

for file in Folder.iterdir():
    if str(file.stem) == str(args.data):
        for video in file.iterdir():
            print (video)
