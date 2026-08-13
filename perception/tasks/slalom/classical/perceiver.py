from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np

from perception.tasks.registry import register_perceiver
from perception.tasks.slalom.classical.detector import PassSide, SlalomDetector
from perception.tasks.TaskPerceiver import TaskPerceiver


@register_perceiver(task="slalom", algo="classical")
class SlalomClassicalPerceiver(TaskPerceiver):
    """Wraps SlalomDetector's hsv/contrast pipe detection to the TaskPerceiver contract."""

    output_class = namedtuple(
        "SlalomOutput", ["target_x", "target_y", "yaw_error", "confidence"]
    )

    def __init__(
        self, pass_side: PassSide = PassSide.LEFT, method: str = "hsv"
    ) -> None:
        super().__init__()
        self.detector = SlalomDetector()
        self.pass_side = pass_side
        self.method = method

    def analyze(
        self, frame: np.ndarray, debug: bool, slider_vals: dict[str, int] | None = None
    ) -> Any:
        estimate = self.detector.detect(
            frame, pass_side=self.pass_side, method=self.method
        )
        result = self.output_class(
            estimate.target_x,
            estimate.target_y,
            estimate.yaw_error,
            estimate.confidence,
        )

        if debug:
            annotated = self.detector.annotate(frame, estimate)
            return result, [annotated]
        return result


if __name__ == "__main__":
    from perception.vis.vis import run

    run(["path/to/video_or_dir"], SlalomClassicalPerceiver(), False)
