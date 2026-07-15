import numpy as np

from perception.tasks.TaskPerceiver import (
    DetectionResult,
    PerceptionOutput,
    TaskContext,
    TaskPerceiver,
)


class LegacySliderAlgo(TaskPerceiver):
    task = "gate"
    algo = "legacy-slider"

    def __init__(self):
        super().__init__(threshold=((0, 255), 100))

    def analyze(self, frame, debug, slider_vals):
        center = (slider_vals["threshold"], frame.shape[0])
        if debug:
            return center, (frame,)
        return center


class RuntimeAlgo(TaskPerceiver):
    task = "slalom"
    algo = "runtime"

    def predict(self, frame, context=None):
        return PerceptionOutput(
            result=DetectionResult(
                task=self.task,
                algo=self.algo,
                target_x=1.0,
                raw={"height": frame.shape[0]},
            ),
            debug_frames={"input": frame},
        )


def test_predict_wraps_legacy_analyze_with_context():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    output = LegacySliderAlgo().predict(
        frame,
        TaskContext(debug=True, tunables={"threshold": 42}, frame_id="frame-1"),
    )

    assert output.result.task == "gate"
    assert output.result.algo == "legacy-slider"
    assert output.result.target_x == 42.0
    assert output.result.target_y == 20.0
    assert output.result.frame_id == "frame-1"
    assert output.result.raw["legacy_result"] == (42, 20)
    assert list(output.debug_frames) == ["frame_0"]


def test_runtime_perceiver_can_define_predict_directly():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    output = RuntimeAlgo().predict(frame)

    assert output.result.task == "slalom"
    assert output.result.algo == "runtime"
    assert output.result.raw == {"height": 10}
    assert output.debug_frames["input"] is frame
