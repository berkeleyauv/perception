from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, Mapping, Optional
import numpy as np


@dataclass
class DetectionResult:
    task: str
    algo: str
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    yaw_error: Optional[float] = None
    confidence: Optional[float] = None
    frame_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackerState:
    task: str
    tracker: str
    active_index: Optional[int] = None
    confidence: Optional[float] = None
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    debug: bool = False
    tunables: Dict[str, int] = field(default_factory=dict)
    frame_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionOutput:
    result: DetectionResult
    debug_frames: Dict[str, np.ndarray] = field(default_factory=dict)


class TaskPerceiver:
    task = "unknown"
    algo = "unknown"

    def __init__(self, **kwargs):
        """Initializes the TaskPerceiver.
        Args:
            kwargs: Each keyworded argument is of the form
                var_name = (range, default_val), where range is the range of values
                for the slider which controls this variable, and default_val is
                the initial value of the slider.
        """
        self.kwargs = kwargs

    def predict(self, frame: np.ndarray, context: Optional[TaskContext] = None) -> PerceptionOutput:
        """Runs perception for runtime consumers.

        New algorithms should override this method. The default implementation
        adapts legacy analyze(frame, debug, slider_vals) implementations so
        vis.py, tests, log replay, and future robot consumers can share the same
        runtime-oriented entrypoint.
        """
        context = context or TaskContext()
        slider_vals = self._slider_values(context.tunables)
        legacy_output = self._call_legacy_analyze(frame, context.debug, slider_vals)

        state = legacy_output
        debug_frames = {}
        if context.debug:
            state, legacy_debug_frames = self._split_legacy_debug_output(legacy_output)
            debug_frames = self._normalize_debug_frames(legacy_debug_frames)

        return PerceptionOutput(
            result=self._legacy_state_to_detection_result(state, context),
            debug_frames=debug_frames,
        )

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]) -> Any:
        """Legacy vis.py-oriented entrypoint.

        Existing algorithms still implement this method. New runtime code should
        call or override predict() instead.

        Args:
            frame: The frame to analyze
            debug: Whether or not to display intermediate images for debugging
            slider_vals: A list of names of the variables which the user should be
                able to control from the Visualizer, mapped to current slider
                value for that variable
        Returns:
            the result of the algorithm
            debug frames must each be same size as original input frame. Might change this in the future.
        """
        raise NotImplementedError("Need to implement with child class.")

    def _call_legacy_analyze(
        self,
        frame: np.ndarray,
        debug: bool,
        slider_vals: Dict[str, int],
    ) -> Any:
        signature = inspect.signature(self.analyze)
        if "slider_vals" in signature.parameters:
            return self.analyze(frame, debug=debug, slider_vals=slider_vals)
        return self.analyze(frame, debug=debug)

    def _slider_values(self, tunables: Optional[Mapping[str, int]]) -> Dict[str, int]:
        slider_vals = {
            name: default_val
            for name, (_slider_range, default_val) in self.kwargs.items()
        }
        if tunables:
            slider_vals.update(tunables)
        return slider_vals

    def _split_legacy_debug_output(self, legacy_output: Any) -> tuple:
        if (
            isinstance(legacy_output, tuple)
            and len(legacy_output) == 2
            and self._looks_like_debug_frames(legacy_output[1])
        ):
            return legacy_output
        return legacy_output, []

    def _looks_like_debug_frames(self, value: Any) -> bool:
        if value is None or isinstance(value, dict):
            return True
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, (list, tuple)):
            return all(self._looks_like_debug_frame(frame) for frame in value)
        return self._looks_like_debug_frame(value)

    def _looks_like_debug_frame(self, value: Any) -> bool:
        return isinstance(value, np.ndarray) or hasattr(value, "canvas")

    def _normalize_debug_frames(self, debug_frames: Any) -> Dict[str, np.ndarray]:
        if debug_frames is None:
            return {}
        if isinstance(debug_frames, dict):
            return debug_frames
        if not isinstance(debug_frames, (list, tuple)):
            debug_frames = [debug_frames]
        return {
            f"frame_{index}": frame
            for index, frame in enumerate(debug_frames)
        }

    def _legacy_state_to_detection_result(
        self,
        state: Any,
        context: TaskContext,
    ) -> DetectionResult:
        target_x = None
        target_y = None
        if (
            isinstance(state, tuple)
            and len(state) >= 2
            and isinstance(state[0], (int, float, np.number))
            and isinstance(state[1], (int, float, np.number))
        ):
            target_x = float(state[0])
            target_y = float(state[1])

        return DetectionResult(
            task=self.task,
            algo=self.algo,
            target_x=target_x,
            target_y=target_y,
            frame_id=context.frame_id,
            raw={"legacy_result": state},
        )
