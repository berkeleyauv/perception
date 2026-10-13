#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_GENERATED_ROOT = (
    Path.home() / "random-background-data-gen/datasets/generated/gate"
)
DEFAULT_OUTPUT = Path("data/gate_yolo_sample")


def newest_dataset_run(generated_root: Path) -> Path:
    runs = sorted(path for path in generated_root.iterdir() if path.is_dir())
    if not runs:
        raise RuntimeError(f"No dataset runs found in {generated_root}")
    return runs[-1]


def split_dataset(
    source: Path,
    output: Path,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    image_dir = source / "images"
    label_dir = source / "labels"
    all_images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_images:
        raise RuntimeError(f"No images found in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(all_images)

    n_images = len(all_images)
    train_end = int(n_images * train_ratio)
    val_end = train_end + int(n_images * val_ratio)
    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:],
    }

    for split_name in splits:
        for subdir in ("images", "labels"):
            split_dir = output / split_name / subdir
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)

    for split_name, images in splits.items():
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            shutil.copy2(image_path, output / split_name / "images" / image_path.name)
            destination_label = output / split_name / "labels" / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, destination_label)
            else:
                destination_label.write_text("")

    return {split_name: len(images) for split_name, images in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Split generated gate data for YOLO.")
    parser.add_argument(
        "--source",
        type=Path,
        help="Dataset folder containing images/ and labels/. Defaults to newest generated gate run.",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=DEFAULT_GENERATED_ROOT,
        help="Root containing generated gate dataset runs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source or newest_dataset_run(args.generated_root)
    print(f"Using: {source}")

    counts = split_dataset(
        source=source,
        output=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    for split_name, count in counts.items():
        print(f"  {split_name}: {count}")
    print(f"\nDone -> {args.output}")


if __name__ == "__main__":
    main()
