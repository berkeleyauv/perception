# Perception Code Overview

## Installation

We will use Conda for managing environments. We recommend installing Miniconda for Python 3.8 [here](https://docs.conda.io/en/latest/miniconda.html).
Then create an environment with

    conda create -n urobotics python3.7

activate it with

    conda activate urobotics

and install all dependencies with

    pip install -r requirements.txt

## misc:
Misc code, camera calibration etc.

## tasks:
Code for specific tasks like 

1. cross: cross detection
1. segmentation:
1. path_marker: path_marker detection
1. spinny_wheel_detection

etc

## vis:
Visualization tools 
Code for testing tasks (Ideally this should be placed a separate folder called `tests`).

## wiki:
Flowchart on TaskPerceiver, TaskReceiver, AlgorithmRunner.