# Perception Code Overview

Code Quality [![CodeFactor](https://www.codefactor.io/repository/github/berkeleyauv/perception/badge)](https://www.codefactor.io/repository/github/berkeleyauv/perception)

## Installation

We use Conda for managing environments. We recommend installing Miniconda [here](https://docs.conda.io/en/latest/miniconda.html).
Then create an environment with

    conda create -n urobotics python=3.11

activate it with

    conda activate urobotics

Then clone the repo in a directory of your choice

    git clone https://github.com/berkeleyauv/perception.git

Change into the cloned repo directory and install it

    pip3 install -e ./


Install all dependencies with

    pip3 install -r requirements-classical.txt

Torch/YOLO/XFeat dependencies are optional and intentionally split from the default install:

    pip3 install -r requirements-torch.txt


Also, our training data is stored here https://www.dropbox.com/sh/rrbfqfutrmifrxs/AAAfXxlcCtWZmUELp4wXyTIxa?dl=0 so download it and unzip it in the same folder as `perception`.

### Cython
To compile archived cythonized code, run the following commands after `cd`ing into the folder with that Cython module's local `setup.py`

    python setup.py build_ext --inplace
    cythonize file_to_cythonize.pyx


## misc:
Misc code, camera calibration etc.

## tasks:
Code for specific tasks is organized by robot task:

1. `gate/classical`: current classical gate detection algorithms.
2. `gate/orientation`: placeholder for the UR-B-Perception orientation port.
3. `slalom/classical`: placeholder for the urb_slalom classical detector port.
4. `buoy`, `torpedo`, `octagon`: scaffolding only.

Older task code is retained under `perception/tasks/_archive`.

In order to create your own algorithm to test:

1. Create <your_algo>.py and put it in one of the specific task folders in perception/tasks.

2. Create a class which extends the TaskPerceiver class. perception/tasks/TaskPerceiver.py includes a template with documentation for how to do this. Register it with the decorator:

        from perception.registry import register_perceiver

        @register_perceiver(task="gate", algo="custom")
        class MyGateAlgo(TaskPerceiver):
            ...

## vis:
Visualization tools 
Code for testing tasks (Ideally this should be placed a separate folder called `tests`).

After writing the code for your specific task algorithm, run it with:

    python -m perception.vis.vis --task gate --algo classical [--data <path to file/directory>] [--profile <function name>] [--save_video]

Use `--compare <algo>` to run a second algorithm from the same task side-by-side:

    python -m perception.vis.vis --task gate --algo classical --compare segmentation_a

Legacy aliases such as `--algorithm gateseg` are still supported temporarily during the consolidation.

## wiki:
Flowchart on TaskPerceiver, TaskReceiver, AlgorithmRunner.
