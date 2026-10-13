import argparse
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = REPO_ROOT / "data/gate_yolo_weights/yolo11n.pt"
DEFAULT_OUTPUT = REPO_ROOT / "data/gate_yolo_predictions"

CLASS_NAMES = {
    0: "gate",
    1: "search_rescue",
    2: "survey_repair"
}

def run(source, model_path, conf, output_dir, show):
    print(f"Model:  {model_path}")
    print(f"Source: {source}")
    print(f"Conf threshold: {conf}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=source,
        conf=conf,
        imgsz=640,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="run",
        exist_ok=True,
        show=show,
    )

    print(f"\nResults saved to {output_dir}/run/")
    print(f"\nDetections summary:")
    for r in results:
        if len(r.boxes) == 0:
            print(f"  {Path(r.path).name}: no detections")
        else:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                name = CLASS_NAMES.get(cls, f"class_{cls}")
                print(f"  {Path(r.path).name}: {name} ({conf_score:.2%} confidence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate detection inference")
    parser.add_argument("--source",  required=True, help="Image, video, folder, or webcam index")
    parser.add_argument("--model",   default=str(DEFAULT_MODEL), help="Path to .pt or .onnx model")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--output",  default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--show",    action="store_true", help="Display results in window")
    args = parser.parse_args()

    run(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        output_dir=args.output,
        show=args.show,
    )
