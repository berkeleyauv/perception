import cv2
import numpy as np

from perception.tasks.slalom.classical import PassSide
from perception.tasks.slalom.classical.perceiver import SlalomClassicalPerceiver


def make_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 70, 65)

    cv2.rectangle(frame, (130, 110), (165, 430), (240, 240, 240), -1)
    cv2.rectangle(frame, (300, 90), (335, 430), (0, 0, 230), -1)
    cv2.rectangle(frame, (475, 130), (510, 430), (240, 240, 240), -1)
    return frame


def test_analyze_without_debug_returns_flat_output() -> None:
    perceiver = SlalomClassicalPerceiver(pass_side=PassSide.LEFT)
    result = perceiver.analyze(make_frame(), debug=False, slider_vals=None)

    assert 0.0 <= result.target_x <= 1.0
    assert 0.0 <= result.target_y <= 1.0
    assert result.confidence > 0.8


def test_analyze_with_debug_returns_annotated_frame_same_size() -> None:
    frame = make_frame()
    perceiver = SlalomClassicalPerceiver(pass_side=PassSide.LEFT)
    result, debug_frames = perceiver.analyze(frame, debug=True, slider_vals=None)

    assert result.confidence > 0.8
    assert len(debug_frames) == 1
    assert debug_frames[0].shape == frame.shape


def test_right_pass_side_targets_right_of_red_pipe() -> None:
    perceiver = SlalomClassicalPerceiver(pass_side=PassSide.RIGHT)
    result = perceiver.analyze(make_frame(), debug=False, slider_vals=None)

    assert 0.5 < result.target_x < 0.75
