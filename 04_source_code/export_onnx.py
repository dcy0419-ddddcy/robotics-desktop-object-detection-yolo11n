from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    best_model = project_root / "03_model" / "best.pt"
    YOLO(best_model).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)


if __name__ == "__main__":
    main()
