#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNS = REPO_ROOT / "data/gate_yolo_runs/detect"
DATAFILE = str(REPO_ROOT / "perception/tasks/gate/yolo/configs/gate_dataset.yaml")
OUT_DIR = REPO_ROOT / "data/gate_yolo_metrics"
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "gate_nano":   RUNS / "gate_nano/weights/best.pt",
    "gate_small":  RUNS / "gate_small/weights/best.pt",
    "gate_medium": RUNS / "gate_medium/weights/best.pt",
}

summary = []
for name, pt in MODELS.items():
    if not pt.exists():
        print(f"[SKIP] {name} - not found")
        continue
    print(f"\n{'='*40}\n{name}")
    model = YOLO(str(pt))
    val   = model.val(data=DATAFILE, split="test", verbose=False)
    info  = model.info(verbose=False)

    m = {
        "model":     name,
        "mAP50":     round(float(val.box.map50), 4),
        "mAP50_95":  round(float(val.box.map),   4),
        "precision": round(float(val.box.mp),     4),
        "recall":    round(float(val.box.mr),     4),
        "params_M":  round(info[0] / 1e6, 3) if info else -1,
        "flops_G":   round(info[1] / 1e9, 3) if info else -1,
    }
    print(json.dumps(m, indent=2))
    (OUT_DIR / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))

    model.export(format="onnx", imgsz=640, opset=12, dynamic=False)
    onnx_src = pt.parent / "best.onnx"
    if onnx_src.exists():
        onnx_src.rename(OUT_DIR / f"{name}.onnx")
        print(f"ONNX -> {OUT_DIR}/{name}.onnx")
    summary.append(m)

if summary:
    csv_path = OUT_DIR / "gate_models_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary -> {csv_path}")
    print(f"\n{'Model':<20} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'Params(M)':>10}")
    for m in sorted(summary, key=lambda x: x["mAP50_95"], reverse=True):
        print(f"{m['model']:<20} {m['mAP50']:>7.4f} {m['mAP50_95']:>9.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['params_M']:>10}")
