# Perception Consolidation Change Map

This file records the structural changes made during the perception consolidation.
It is meant to help reviewers distinguish source moves from new code, archives,
and generated artifacts.

## Core Interface Changes

| File | Change |
| --- | --- |
| `perception/tasks/TaskPerceiver.py` | Added runtime-facing dataclasses: `DetectionResult`, `TrackerState`, `TaskContext`, and `PerceptionOutput`. Added `TaskPerceiver.predict(frame, context)` while keeping legacy `analyze(...)` support. |
| `perception/registry.py` | Added decorator-based perceiver registration with `@register_perceiver(task=..., algo=...)`, discovery, and lookup helpers. |
| `perception/__init__.py` | Replaced manual algorithm imports/`ALGOS` export with thin registry helper exports. |
| `perception/vis/vis.py` | Updated visualization to consume `predict(...)`, added `--task`, `--algo`, and `--compare`, and kept temporary legacy `--algorithm` aliases. |

## Moved Active Gate Code

| Old Path | New Path | Notes |
| --- | --- | --- |
| `perception/tasks/gate/GateCenterAlgo.py` | `perception/tasks/gate/classical/GateCenterAlgo.py` | Registered as `task="gate", algo="classical"`. |
| `perception/tasks/gate/GateSegmentationAlgoA.py` | `perception/tasks/gate/classical/GateSegmentationAlgoA.py` | Registered as `task="gate", algo="segmentation_a"`. |
| `perception/tasks/gate/GateSegmentationAlgoB.py` | `perception/tasks/gate/classical/GateSegmentationAlgoB.py` | Registered as `task="gate", algo="segmentation_b"`. |
| `perception/tasks/gate/GateSegmentationAlgoC.py` | `perception/tasks/gate/classical/GateSegmentationAlgoC.py` | Registered as `task="gate", algo="segmentation_c"`. |
| `perception/tasks/gate/archive/` | `perception/tasks/gate/classical/archive/` | Kept with the gate classical code it supports. |

## Added Slalom Code

| Path | Source | Notes |
| --- | --- | --- |
| `perception/tasks/slalom/classical/slalom_detector.py` | `harshi-puli/urb-slalom/src/urb_slalom/detector.py` | Ported classical HSV/contrast slalom detector. Added `SlalomClassicalPerceiver` wrapper registered as `task="slalom", algo="classical"`. |
| `perception/tasks/slalom/slalom_sequence_tracker.py` | `harshi-puli/urb-slalom/src/urb_slalom/gate_sequence_tracker.py` | Renamed/relocated as slalom-specific tracker. Keeps original behavior and exposes `GateSequenceTracker = SlalomSequenceTracker` compatibility alias. |
| `tests/slalom/test_detector.py` | `harshi-puli/urb-slalom/tests/test_detector.py` | Ported detector regression tests with consolidated import paths. |
| `perception/tasks/slalom/yolo/` | New scaffold | Placeholder only; no slalom YOLO model exists to port. |

## Added Gate Orientation Code

| Path | Source | Notes |
| --- | --- | --- |
| `perception/tasks/gate/orientation/classical_orientation.py` | `raymondt31/UR-B-Perception/ClassicalOrientation.py` | Ported LAB/ROI orientation refinement. Added `ClassicalOrientationPerceiver`, registered as `task="gate", algo="classical_orientation"`. Expects `gate_roi` or `box` in `TaskContext.metadata`. |
| `perception/tasks/gate/orientation/xfeat_orientation.py` | `raymondt31/UR-B-Perception/XFeatOrientation.py` | Ported XFeat homography alignment. Added `XFeatOrientationPerceiver`, registered as `task="gate", algo="xfeat_orientation"`. Torch/XFeat imports are lazy/optional. |
| `perception/tasks/gate/orientation/modules/` | `raymondt31/UR-B-Perception/modules/` | Copied XFeat support modules and patched runtime imports to consolidated package paths. |
| `perception/tasks/gate/orientation/Gate1.png` | `raymondt31/UR-B-Perception/Gate1.png` | Reference image asset for XFeat orientation. |
| `perception/tasks/gate/orientation/Gate2.png` | `raymondt31/UR-B-Perception/Gate2.png` | Reference image asset from source repo. |
| `perception/tasks/gate/yolo/` | New scaffold | Placeholder until a current gate YOLO model is confirmed. |

## Archived Legacy Task Code

These files were not intended to be deleted permanently. They were moved under
`perception/tasks/_archive/` so active task code can live in the new consolidated
layout.

| Old Path | New Path |
| --- | --- |
| `perception/tasks/cross/` | `perception/tasks/_archive/cross/` |
| `perception/tasks/dice/` | `perception/tasks/_archive/dice/` |
| `perception/tasks/path_marker/` | `perception/tasks/_archive/path_marker/` |
| `perception/tasks/roulette/` | `perception/tasks/_archive/roulette/` |
| `perception/tasks/segmentation/` | `perception/tasks/_archive/segmentation/` |
| `perception/tasks/slot_machine/` | `perception/tasks/_archive/slot_machine/` |
| `perception/tasks/slots/` | `perception/tasks/_archive/slots/` |
| `perception/tasks/sanity_test.py` | `tests/sanity_test.py` |

## Archived Visual Test Code

| Old Path | New Path |
| --- | --- |
| `perception/vis/TestAlgo.py` | `perception/tasks/_examples/test_algo.py` |
| `perception/vis/TestTasks/BackgroundRemoval.py` | `perception/_archive/TestTasks/BackgroundRemoval.py` |
| `perception/vis/TestTasks/DepthMap.py` | `perception/_archive/TestTasks/DepthMap.py` |

## Added Empty Task Scaffolds

These packages were added as structure only. They do not contain implemented
detectors yet.

- `perception/tasks/buoy/`
- `perception/tasks/torpedo/`
- `perception/tasks/octagon/`

## Packaging, Dependencies, and CI

| File | Change |
| --- | --- |
| `setup.py` | Removed in favor of `pyproject.toml`. |
| `pyproject.toml` | Added setuptools build config, Python `>=3.11`, package metadata, dependencies, and dev/torch optional dependency groups. |
| `requirements.txt` | Changed to include `requirements-classical.txt`. |
| `requirements-classical.txt` | Added pinned classical/runtime/dev dependencies. |
| `requirements-torch.txt` | Added optional Torch/YOLO/XFeat dependency set. |
| `.github/workflows/ci.yml` | Added GitHub Actions lint and classical test jobs on Python 3.11. |
| `README.md` | Updated install commands, Python version, registry usage, and `vis.py` examples. |

## Generated Artifacts To Watch

The Cython rebuild created platform-specific generated outputs under:

- `perception/tasks/_archive/segmentation/saliency_detection/*.cpython-310-darwin.so`
- regenerated `*.c` files in the same folder

These are build artifacts from the current local Python 3.10 environment. They
should only be committed if the team intentionally tracks compiled/generated
Cython outputs. Otherwise, leave them ignored or remove them before a final PR.

Python `__pycache__/` folders are also generated artifacts and are not part of
the intended source changes.

## Verification Performed

- Python compile checks on moved/ported source files passed.
- `python -m perception.vis.vis --help` passed.
- Registry lookups passed for:
  - `slalom/classical`
  - `gate/classical`
  - `gate/classical_orientation`
  - `gate/xfeat_orientation`
- Slalom detector regression tests were run directly and passed.
- Archived Cython extension build completed in the active Python 3.10 env.

## Verification Still Needed

- Fresh Python 3.11 environment install.
- Full `pytest` run.
- Runtime import check for archived Cython saliency module after installing `scipy`.
- XFeat orientation runtime check in the optional Torch dependency environment.
