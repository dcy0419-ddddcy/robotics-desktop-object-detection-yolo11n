import time
from pathlib import Path

import cv2
from ultralytics import YOLO


CAMERA_INDEX = 1
CONFIDENCE = 0.25
DEVICE = "cpu"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "03_model" / "best.pt"
    model = YOLO(model_path)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise RuntimeError("无法打开外置 USB 摄像头，请检查设备是否被其他程序占用")

    previous_time = time.perf_counter()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("摄像头画面读取失败")

            result = model.predict(
                frame,
                imgsz=640,
                conf=CONFIDENCE,
                device=DEVICE,
                verbose=False,
            )[0]
            display = result.plot()

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            cv2.putText(display, f"FPS: {fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Desktop Object Detection - press Q to exit", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
