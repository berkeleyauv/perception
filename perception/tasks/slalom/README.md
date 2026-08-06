# slalom

Perception algorithms for RoboSub Task 2: Avoid Debris / Slalom.

The task has three repeated pipe sets arranged as:

```text
WHITE    RED    WHITE
```

The AUV should navigate through each set while keeping the red pipe on the same side it used when passing the gate, and while staying vertically within the pipe area.

## Structure

```text
slalom/
├── classical/                  # OpenCV detector for red/white vertical PVC pipes (hsv + contrast segmentation)
├── slalom_sequence_tracker.py  # Multi-gate state across frames: side consistency, distance, and search bias
├── yolo/                       # Placeholder — no YOLO-based detector exists yet
└── tests/                      # Synthetic unit tests that exercise the classical detector without pool footage
```

`classical/` is ported from the original `urb-slalom` prototype sandbox. That sandbox's CLI and video/GoPro test scripts were not carried over here — this package exposes the detector and tracker as a plain Python API instead.

## Usage

```python
import cv2

from perception.tasks.slalom.classical import PassSide, SlalomDetector
from perception.tasks.slalom.slalom_sequence_tracker import SlalomSequenceTracker

frame = cv2.imread("path/to/frame.png")

detector = SlalomDetector()
estimate = detector.detect(frame, pass_side=PassSide.LEFT)
annotated = detector.annotate(frame, estimate)

# Optional: track gate-passing state across a video's frames
tracker = SlalomSequenceTracker(focal_length_px=900.0)
gate_result = tracker.update(estimate, image_width=frame.shape[1])
```

Use `detector.detect_white_pole_target(...)` / `detector.annotate_white_pole(...)` for proxy footage that has a single white pole instead of a full red/white pipe set. Pass `method="contrast"` to `detect(...)` for footage where water attenuation makes the red pipe read as near-black rather than red-hued (see Known Limitation below).

## Test Data

`Slalom Task/` is a labeled Roboflow dataset (`train/`, `valid/`, `test/`, each with `images/` + YOLO-format `labels/`, plus `data.yaml` with classes `red`/`white`) — useful for testing detection against real course footage rather than just synthetic frames. `crc-photos/` holds additional still frames from pool/tank footage.

## Run Tests

```bash
pytest perception/tasks/slalom/tests
```

## Algorithm

1. Convert the frame to HSV.
2. Segment red pipes with two hue ranges, because red wraps around HSV hue 0.
3. Segment white pipes using low saturation and high value.
4. Clean masks with morphological open/close operations.
5. Extract contours and keep tall, vertical, pipe-like bounding boxes.
6. Pick the best red pipe and the nearest white pipe on each side.
7. Compute a target point in the open corridor that keeps the red pipe on the requested side.

The detector returns both raw pipe detections and a `SlalomEstimate`:

- `target_x`: normalized horizontal steering target, where `0.0` is left and `1.0` is right.
- `target_y`: normalized vertical target, useful for staying within the pipe height.
- `yaw_error`: normalized horizontal error from image center.
- `confidence`: simple confidence score based on whether the red pipe and adjacent white pipe were detected.

### Known Limitation: Color Cast

Pixel sampling against both the `Slalom Task` dataset and our own GoPro footage shows that at typical course range, water attenuation shifts the red pipe's hue to nearly match the white pipe's and the background's (within a few degrees of hue) — only brightness (HSV Value) reliably separates them. The current hue-based `DetectorConfig` thresholds do not account for this and should be expected to under-detect on real footage until segmentation is reworked around relative brightness rather than hue.

## ROS 2 Porting Notes

The next step is to wrap `SlalomDetector` in a ROS 2 node that:

- subscribes to `/camera/usb/front/compressed`
- publishes debug masks/images
- publishes raw `CVObject` pipe detections
- publishes one high-level `CVObject` target for task planning

Suggested topics:

```text
/cv/front/slalom_red_pipe/bounding_box
/cv/front/slalom_white_left/bounding_box
/cv/front/slalom_white_right/bounding_box
/cv/front/slalom_target/bounding_box
```