# urb-slalom

Prototype perception algorithms for RoboSub Task 2: Avoid Debris / Slalom.

The task has three repeated pipe sets arranged as:

```text
WHITE    RED    WHITE
```

The AUV should navigate through each set while keeping the red pipe on the same side it used when passing the gate, and while staying vertically within the pipe area.

## What This Repo Contains

- A classical OpenCV detector for red and white vertical PVC pipes.
- A high-level slalom target estimator that converts pipe detections into a steering target.
- A CLI for testing the detector on images or video.
- Synthetic unit tests that exercise the core detector without needing pool footage.

This is meant to be a fast iteration sandbox before porting the algorithm into the ROS 2 `cv` package.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run On An Image

```bash
urb-slalom path/to/frame.png --show
```

Write an annotated result:

```bash
urb-slalom path/to/frame.png --output annotated.png
```

Use the opposite pass side:

```bash
urb-slalom path/to/frame.png --red-side left
```

## Run On A Directory Of Images

Point the CLI at a folder instead of a single file to browse a whole dataset. It prints an estimate per image, and with `--show` pops up a window, pausing on each frame until you press a key (`q`/Esc quits early).

```bash
# Live popup, one keypress per image
urb-slalom "Slalom Task/test/images" --show

# Or write annotated copies to inspect later, no popup needed
urb-slalom "Slalom Task/test/images" --output outputs/annotated_test
```

`Slalom Task/` is a labeled Roboflow dataset (`train/`, `valid/`, `test/`, each with `images/` + YOLO-format `labels/`, plus `data.yaml` with classes `red`/`white`) — useful for testing detection against real course footage rather than just synthetic frames.

## Test Proxy Footage With One White Pole

If you have course footage that is not true slalom footage, but has a similar vertical white pole, use `white-pole` mode. This mode ignores the full red/white slalom layout and simply places a yellow target dot to the left or right of the detected white pole.

```bash
mkdir -p outputs
urb-slalom "GOPR1146 copy.MP4" --mode white-pole --target-side left --output outputs/gopro_white_pole_left.mp4
```

Or use the helper script:

```bash
./scripts/test_gopro_white_pole.sh
```

The annotated output draws:

- a white bounding box around the detected white pole
- a yellow dot on the requested side of the pole
- a yellow line from the bottom-center of the frame to the target dot

This is only a proxy test. It is useful for tuning white-pole segmentation and target placement, but it does not validate the full slalom behavior because true slalom needs red and white pipe-set reasoning.

## Run Tests

```bash
pytest
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