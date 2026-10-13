# Gate YOLO

Utilities for training, evaluating, and running YOLO gate detection models.

## Layout

- `configs/gate_dataset.yaml`: YOLO dataset config for the gate classes.
- `configs/legacy_dataset.yaml`: Older dataset config kept for reference.
- `split_gate_data.py`: Splits a generated `images/` + `labels/` dataset into YOLO `train/`, `val/`, and `test/` folders.
- `predict_gate.py`: Runs inference with an exported or PyTorch YOLO model.
- `export_metrics.py`: Evaluates trained models and exports metrics/ONNX artifacts.

Local sample data and model weights live under the repo-level ignored `data/` folder:

- `data/gate_yolo_sample/`
- `data/gate_yolo_weights/`
