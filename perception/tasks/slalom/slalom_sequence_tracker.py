"""
SlalomSequenceTracker
--------------------
Wraps classical.detector.SlalomDetector to add the pieces the RoboSub
slalom task scores that a single-frame detect() call doesn't give you:

  1. Multi-gate state — which of the 3 gates you're approaching, and
     confirmation that you've passed one before advancing to the next.
  2. Side consistency — remembers which side you passed the red pipe on
     at gate 1, flags gate 2/3 if they'd put it on the other side.
  3. Distance/depth cue — approximate real-world range to the current gate
     from its known 1524mm (60in) white-to-white width vs. the pixel span
     your detector already computes between left_white_pipe/right_white_pipe.
  4. Stagger-informed bias — after a gate is passed, the next gate sits
     ~1000mm laterally offset (per the course drawing), alternating side.
     Useful to bias where you point the camera/search first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.tasks.slalom.classical.detector import PassSide, SlalomEstimate

# --- Known course geometry (mm), from the design drawing -------------------
GATE_WIDTH_MM = 1524.0  # white-to-white pipe center distance
GATE_STAGGER_MM = 1000.0  # lateral offset of middle gate from outer gates
GATE1_TO_GATE3_MM = 2002.9  # longitudinal spacing, gate 1 to gate 3 centerline


class GateIndex(Enum):
    GATE_1 = 0
    GATE_2 = 1
    GATE_3 = 2
    DONE = 3


@dataclass
class GateResult:
    index: GateIndex
    pass_side: PassSide | None
    distance_mm: float | None
    confidence: float


@dataclass
class TrackerState:
    current_gate: GateIndex = GateIndex.GATE_1
    committed_side: PassSide | None = None
    side_mismatches: int = 0
    gates_passed: int = 0
    _was_valid_last_frame: bool = False
    _last_red_center_x: float | None = None


class SlalomSequenceTracker:
    """
    Call update(estimate, image_width) once per frame with the SlalomEstimate
    SlalomDetector.detect() already returns. Everything below reuses fields
    that already exist on SlalomEstimate/PipeDetection — no changes needed
    to detector.py itself.
    """

    def __init__(self, focal_length_px: float):
        # From camera calibration (fx). If uncalibrated, derive empirically:
        # fx = (pixel_width_at_known_range_mm * range_mm) / GATE_WIDTH_MM
        # using a frame where you know the real distance to a gate.
        self.focal_length_px = focal_length_px
        self.state = TrackerState()

    def estimate_distance_mm(self, estimate: SlalomEstimate) -> float | None:
        if estimate.left_white_pipe is None or estimate.right_white_pipe is None:
            return None
        pixel_span = abs(
            estimate.right_white_pipe.center_x - estimate.left_white_pipe.center_x
        )
        if pixel_span <= 0:
            return None
        return (GATE_WIDTH_MM * self.focal_length_px) / pixel_span

    def update(self, estimate: SlalomEstimate, image_width: int) -> GateResult:
        distance_mm = self.estimate_distance_mm(estimate)
        result = GateResult(
            index=self.state.current_gate,
            pass_side=None,
            distance_mm=distance_mm,
            confidence=estimate.confidence,
        )

        # A gate that was tracked (valid) and just stopped being valid means
        # it left the frame — count it as passed and infer which side the
        # red pipe was on from the last frame it was still visible.
        gate_just_exited = (
            self.state._was_valid_last_frame
            and not estimate.valid
            and self.state._last_red_center_x is not None
        )

        if gate_just_exited and self.state.current_gate != GateIndex.DONE:
            side = (
                PassSide.LEFT
                if self.state._last_red_center_x < image_width / 2
                else PassSide.RIGHT
            )
            result.pass_side = side

            if self.state.committed_side is None:
                self.state.committed_side = side
            elif side != self.state.committed_side:
                self.state.side_mismatches += 1

            self.state.gates_passed += 1
            self.state.current_gate = GateIndex(
                min(self.state.current_gate.value + 1, GateIndex.DONE.value)
            )

        self.state._was_valid_last_frame = estimate.valid
        if estimate.valid and estimate.red_pipe is not None:
            self.state._last_red_center_x = estimate.red_pipe.center_x

        return result

    def expected_search_bias_px(self, px_per_mm: float) -> float:
        """
        Signed pixel offset to bias next-frame search toward the expected
        1000mm stagger, alternating direction gate to gate, pointing back
        toward the course centerline relative to the committed pass side.
        """
        if self.state.gates_passed == 0 or self.state.committed_side is None:
            return 0.0
        direction = 1.0 if self.state.committed_side == PassSide.LEFT else -1.0
        alternate = 1 if self.state.gates_passed % 2 == 1 else -1
        return direction * alternate * GATE_STAGGER_MM * px_per_mm

    @property
    def run_summary(self) -> dict:
        return {
            "gates_passed": self.state.gates_passed,
            "committed_side": (
                self.state.committed_side.value if self.state.committed_side else None
            ),
            "side_mismatches": self.state.side_mismatches,
            "side_consistent": self.state.side_mismatches == 0,
        }
