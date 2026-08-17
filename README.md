# Perception Code Overview

Code Quality [![CodeFactor](https://www.codefactor.io/repository/github/berkeleyauv/perception/badge)](https://www.codefactor.io/repository/github/berkeleyauv/perception)

## Installation

We use [uv](https://docs.astral.sh/uv/) for managing both the Python version and the virtual environment. Install it once per machine:

    curl -LsSf https://astral.sh/uv/install.sh | sh

(or `brew install uv` on macOS). Full install docs [here](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repo in a directory of your choice

    git clone https://github.com/berkeleyauv/perception.git
    cd perception

Create the virtual environment — this downloads and pins Python 3.11 (required to support YOLO models) if you don't already have it, and creates `.venv/` in the repo:

    uv venv --python 3.11

Activate it

    source .venv/bin/activate

Install the package in editable mode

    uv pip install -e .

Install dependencies with

    uv pip install -r requirements-classical.txt

If you also need YOLO/torch-based models, install the torch group instead (it includes everything in `requirements-classical.txt`, plus `torch`, `torchvision`, and `ultralytics`)

    uv pip install -r requirements-torch.txt


Also, our training data is stored here https://www.dropbox.com/sh/rrbfqfutrmifrxs/AAAfXxlcCtWZmUELp4wXyTIxa?dl=0 so download it and unzip it in the same folder as `perception`.

### Cython
To compile cythonized code, run the following commands (with the venv activated) after `cd`ing into the folder with Cython `setup.py`

    python setup.py build_ext --inplace
    cythonize file_to_cythonize.pyx


## misc:
Misc code, camera calibration etc.

## tasks:
Code for specific competition tasks, one folder per task:

1. `gate`: qualification gate detection (`classical/`) and orientation estimation (`orientation/`)
2. `slalom`: slalom pipe-set detection (`classical/`) and pipe-set sequence tracking (`slalom_sequence_tracker.py`)
3. `path_marker`: path marker detection
4. `buoy`, `torpedo`, `octagon`: scaffolded, no detection logic yet
5. `_archive`: retired tasks (cross, dice, roulette, slots) kept for reference, excluded from registry discovery

In order to create your own algorithm to test:

1. Create `<your_algo>.py` and put it in the relevant task folder under `perception/tasks/` (e.g. a new classical approach goes in that task's `classical/`).

2. Create a class which extends `TaskPerceiver` (see `perception/tasks/TaskPerceiver.py` for the template with documentation) and decorate it with `@register_perceiver(task=..., algo=...)` from `perception/tasks/registry.py`. This is what makes it discoverable by `vis.py` — see the **vis** section below.

## vis:
Visualization tools for interactively running and debugging task algorithms.

Every algorithm is a `TaskPerceiver` subclass (see `perception/tasks/TaskPerceiver.py`) decorated with `@register_perceiver(task=..., algo=...)`. Decorating a class is all that's needed to make it discoverable — there's no shared file to hand-edit:

    from perception.tasks.registry import register_perceiver
    from perception.tasks.TaskPerceiver import TaskPerceiver

    @register_perceiver(task="gate", algo="my_algo")
    class MyAlgo(TaskPerceiver):
        ...

Then run it with:

    python -m perception.vis.vis --task gate --algo my_algo [--data <path to file/directory>] [--profile <function name>] [--save_video] [--resize <scale>]

- `--task` / `--algo` are required and select the registered perceiver to run.
- `--data` defaults to your webcam; point it at an image, video, or a directory of either.
- `--profile` is off by default; pass a `cProfile` stats key (or omit for `'all'`) to profile the run.
- `--save_video` writes the debug-frame grid to `vis_rec.mp4`.
- `--resize` scales every frame before display (default `1.0`, no resize).
- `--compare <algo>` runs a second algo for the same `--task` on the same frames and stacks it below the primary algo's grid in one "Debug Frames" window, each half labeled with its algo name in the top-left corner — useful for A/B'ing two algorithms (e.g. `center` vs. `segmentation_a` for `gate`) against the same footage. Stacking below (rather than beside) keeps each pane's width unchanged, so sub-frame resolution and label/slider text stay legible regardless of how many debug frames either algo returns. Sliders for both algos appear in the same window, prefixed with their algo name (e.g. `center: canny_low`) to keep them distinguishable. `--save_video` saves the combined, labeled view.

While a window is focused: `q`/`Esc` quits, `p` pauses, `i`/`o` slow down/speed up frame playback.

## wiki:
Flowchart on TaskPerceiver, TaskReceiver, AlgorithmRunner.
