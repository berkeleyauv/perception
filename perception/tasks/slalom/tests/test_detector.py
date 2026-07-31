import cv2
import numpy as np
from perception.tasks.slalom.classical import PassSide, SlalomDetector


def make_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 70, 65)

    cv2.rectangle(frame, (130, 110), (165, 430), (240, 240, 240), -1)
    cv2.rectangle(frame, (300, 90), (335, 430), (0, 0, 230), -1)
    cv2.rectangle(frame, (475, 130), (510, 430), (240, 240, 240), -1)
    return frame


def test_detects_pipe_triplet() -> None:
    detector = SlalomDetector()
    estimate = detector.detect(make_frame(), pass_side=PassSide.LEFT)

    assert estimate.valid
    assert estimate.red_pipe is not None
    assert estimate.left_white_pipe is not None
    assert estimate.right_white_pipe is not None
    assert estimate.confidence > 0.8


def test_left_pass_side_targets_left_corridor() -> None:
    detector = SlalomDetector()
    estimate = detector.detect(make_frame(), pass_side=PassSide.LEFT)

    assert estimate.target_x < estimate.red_pipe.center_x / 640
    assert 0.25 < estimate.target_x < 0.5


def test_right_pass_side_targets_right_corridor() -> None:
    detector = SlalomDetector()
    estimate = detector.detect(make_frame(), pass_side=PassSide.RIGHT)

    assert estimate.target_x > estimate.red_pipe.center_x / 640
    assert 0.5 < estimate.target_x < 0.75


def test_missing_white_pipe_uses_red_pipe_fallback() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 70, 65)
    cv2.rectangle(frame, (300, 90), (335, 430), (0, 0, 230), -1)

    detector = SlalomDetector()
    estimate = detector.detect(frame, pass_side=PassSide.LEFT)

    assert estimate.valid
    assert estimate.confidence < 0.5
    assert estimate.target_x < estimate.red_pipe.center_x / 640


def test_white_pole_mode_targets_left_of_pole() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 70, 65)
    cv2.rectangle(frame, (300, 90), (335, 430), (240, 240, 240), -1)

    detector = SlalomDetector()
    estimate = detector.detect_white_pole_target(frame, target_side=PassSide.LEFT)

    assert estimate.valid
    assert estimate.white_pipe is not None
    assert estimate.target_x < estimate.white_pipe.center_x / 640


def test_white_pole_mode_targets_right_of_pole() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 70, 65)
    cv2.rectangle(frame, (300, 90), (335, 430), (240, 240, 240), -1)

    detector = SlalomDetector()
    estimate = detector.detect_white_pole_target(frame, target_side=PassSide.RIGHT)

    assert estimate.valid
    assert estimate.white_pipe is not None
    assert estimate.target_x > estimate.white_pipe.center_x / 640


def test_white_pole_auto_side_targets_toward_image_center() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 70, 65)
    cv2.rectangle(frame, (120, 90), (155, 430), (240, 240, 240), -1)

    detector = SlalomDetector()
    estimate = detector.detect_white_pole_target(frame, target_side=None)

    assert estimate.valid
    assert estimate.white_pipe is not None
    assert estimate.target_side == PassSide.RIGHT
    assert estimate.target_x > estimate.white_pipe.center_x / 640


def test_white_pole_mode_targets_midpoint_when_dark_red_pair_is_visible() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (60, 120, 120)
    cv2.rectangle(frame, (170, 120), (185, 390), (245, 245, 245), -1)
    cv2.line(frame, (420, 90), (455, 410), (15, 25, 25), 12)

    detector = SlalomDetector()
    estimate = detector.detect_white_pole_target(
        frame, target_side=None, method="contrast"
    )

    assert estimate.valid
    assert estimate.white_pipe is not None
    assert estimate.red_pipe is not None

    expected_midpoint = (
        (estimate.white_pipe.center_x + estimate.red_pipe.center_x) / 2 / 640
    )
    assert abs(estimate.target_x - expected_midpoint) < 0.02
