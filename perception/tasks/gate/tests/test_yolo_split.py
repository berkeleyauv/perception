from pathlib import Path

from perception.tasks.gate.yolo.split_gate_data import split_dataset


def write_sample(source: Path, name: str) -> None:
    (source / "images" / f"{name}.jpg").write_bytes(b"fake image")
    (source / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_split_dataset_creates_yolo_train_val_test_layout(tmp_path: Path) -> None:
    source = tmp_path / "generated"
    output = tmp_path / "split"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()

    for index in range(10):
        write_sample(source, f"sample_{index:05d}")

    counts = split_dataset(source, output, train_ratio=0.6, val_ratio=0.2, seed=7)

    assert counts == {"train": 6, "val": 2, "test": 2}
    for split_name, expected_count in counts.items():
        images = list((output / split_name / "images").glob("*.jpg"))
        labels = list((output / split_name / "labels").glob("*.txt"))
        assert len(images) == expected_count
        assert len(labels) == expected_count


def test_split_dataset_writes_empty_label_when_missing(tmp_path: Path) -> None:
    source = tmp_path / "generated"
    output = tmp_path / "split"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    (source / "images" / "sample_00000.jpg").write_bytes(b"fake image")

    split_dataset(source, output, train_ratio=1.0, val_ratio=0.0)

    assert (output / "train" / "labels" / "sample_00000.txt").read_text() == ""
