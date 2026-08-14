import cv2
import numpy as np

from perception.tasks.path_marker.perceiver import PathMarkerPerceiver


def make_frame() -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (40, 60, 50)
    cv2.line(frame, (80, 60), (160, 120), (230, 230, 230), 4)
    cv2.line(frame, (160, 120), (80, 180), (230, 230, 230), 4)
    return frame


def test_analyze_without_debug_returns_angle_pair() -> None:
    perceiver = PathMarkerPerceiver()
    result = perceiver.analyze(make_frame(), debug=False, slider_vals=None)

    assert hasattr(result, "bottom_angle")
    assert hasattr(result, "top_angle")
    assert result.bottom_angle is None or isinstance(result.bottom_angle, float)
    assert result.top_angle is None or isinstance(result.top_angle, float)


def test_analyze_with_debug_returns_frame_matching_input_size() -> None:
    frame = make_frame()
    perceiver = PathMarkerPerceiver()
    result, debug_frames = perceiver.analyze(frame, debug=True, slider_vals=None)

    assert hasattr(result, "bottom_angle")
    assert len(debug_frames) == 1
    assert debug_frames[0].shape[:2] == frame.shape[:2]
