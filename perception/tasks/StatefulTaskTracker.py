from dataclasses import dataclass, field
from typing import Any

from perception.tasks.TaskPerceiver import DetectionResult, TrackerState


@dataclass
class TrackerContext:
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StatefulTaskTracker:
    task = "unknown"
    tracker = "unknown"

    def reset(self) -> None:
        """Reset any state accumulated across frames."""

    def update(
        self,
        detection: DetectionResult,
        context: TrackerContext | None = None,
    ) -> TrackerState:
        raise NotImplementedError("Need to implement with child class.")
