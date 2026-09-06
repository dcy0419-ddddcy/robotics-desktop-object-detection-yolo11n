from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = YOLO(project_root / "03_model" / "best.pt")
    data = project_root / "02_dataset" / "four_class_dataset" / "data.yaml"

    model.val(
        data=data,
        split="test",
        imgsz=640,
        batch=4,
        device="cpu",
        plots=True,
        project=project_root / "05_results",
        name="reproduced_evaluation",
        exist_ok=True,
    )
    model.predict(
        source=project_root / "02_dataset" / "four_class_dataset" / "dataset" / "images" / "test",
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        save_txt=True,
        save_conf=True,
        project=project_root / "05_results",
        name="reproduced_predictions",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
