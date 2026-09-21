from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_IMAGES = REPO_ROOT / "data/gate_yolo_sample/test/images"
DEFAULT_MODEL = REPO_ROOT / "data/gate_yolo_weights/yolo11n.pt"


def test_yolo_model_predicts_on_sample_gate_images(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    if not DEFAULT_MODEL.exists():
        pytest.skip(f"Missing local YOLO weights: {DEFAULT_MODEL}")
    if not SAMPLE_IMAGES.exists():
        pytest.skip(f"Missing local sample images: {SAMPLE_IMAGES}")

    image_paths = sorted(SAMPLE_IMAGES.glob("*.jpg"))[:2]
    if len(image_paths) < 2:
        pytest.skip("Need at least two gate sample images for YOLO smoke test")

    model = YOLO(str(DEFAULT_MODEL))
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=0.25,
        imgsz=640,
        project=str(tmp_path),
        name="predict",
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
    )

    assert len(results) == 2
    assert (tmp_path / "predict").exists()
