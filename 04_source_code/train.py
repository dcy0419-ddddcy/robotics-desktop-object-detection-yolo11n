from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = YOLO("yolo11n.pt")
    model.train(
        data=project_root / "02_dataset" / "four_class_dataset" / "data.yaml",
        epochs=100,
        patience=20,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        seed=20260831,
        deterministic=True,
        project=project_root / "05_results" / "reproduced_training",
        name="desktop4_yolo11n",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
