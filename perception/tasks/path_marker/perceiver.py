from __future__ import annotations

from collections import namedtuple
from typing import Any

import cv2
import numpy as np

from perception.tasks.path_marker.path_marker_detection import find_path_marker
from perception.tasks.registry import register_perceiver
from perception.tasks.segmentation.combinedFilter import init_combined_filter
from perception.tasks.TaskPerceiver import TaskPerceiver


@register_perceiver(task="path_marker", algo="classical")
class PathMarkerPerceiver(TaskPerceiver):
    """Wraps find_path_marker's Hough-line angle detection to the TaskPerceiver contract."""

    output_class = namedtuple("PathMarkerOutput", ["bottom_angle", "top_angle"])

    def __init__(self) -> None:
        super().__init__()
        self.combined_filter = init_combined_filter()

    def analyze(
        self, frame: np.ndarray, debug: bool, slider_vals: dict[str, int] | None = None
    ) -> Any:
        threshed = self.combined_filter(frame, display_figs=False)
        angles = find_path_marker(threshed, draw_figs=False)

        if angles is None:
            result = self.output_class(None, None)
        else:
            bot_angle, top_angle = angles
            result = self.output_class(bot_angle, top_angle)

        if debug:
            debug_frame = threshed[:, :, 0] if threshed.ndim == 3 else threshed
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
            return result, [debug_frame]
        return result


if __name__ == "__main__":
    from perception.vis.vis import run

    run(["path/to/video_or_dir"], PathMarkerPerceiver(), False)
