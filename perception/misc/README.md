# Miscellaneous Perception Experiments

This directory contains older exploratory scripts, notebooks, calibration utilities, and experimental image-processing code that has not been promoted into a task-specific package.

## Current Contents

- `dark_channel/`: dark-channel dehazing and depth-map experiments used by background-removal exploration.
- `combined_filter.py` and `combinedFilTest.py`: earlier combined-filter experiments. Active task code currently uses `perception.tasks.segmentation.combinedFilter`.
- `camera_chessboard_calibration.py`: camera calibration utility.
- `optical_flow.py`, `featureGray2_higher_order_fns.py`, and notebooks: exploratory perception prototypes.
- `DiceLabels.csv`: label data for older dice detection experiments.

## Guidelines

- New production task code should live under `perception/tasks/<task>/`.
- Shared reusable image-processing code should live under `perception/tasks/segmentation/` or another task-owned module with tests.
- Keep notebooks and one-off experiments here only when they are still useful reference material.
- If a misc script becomes part of a task workflow, move it into that task package and update imports/tests in the same change.

