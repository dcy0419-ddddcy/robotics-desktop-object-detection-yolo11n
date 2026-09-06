import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def synchronize(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a YOLO model on Jetson camera frames.")
    parser.add_argument("--model", required=True, help="Path to best.pt, best.onnx, or best.engine")
    parser.add_argument("--source", default="/dev/video0", help="Camera index, device, or video path")
    parser.add_argument("--device", default="0", help="Ultralytics device, normally 0 on Jetson")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="outputs/benchmark/jetson_camera.csv")
    args = parser.parse_args()

    model = YOLO(args.model)
    camera = cv2.VideoCapture(parse_source(args.source))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera source: {args.source}")

    records = []
    try:
        for frame_id in range(args.warmup + args.frames):
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Camera frame acquisition failed")

            synchronize(args.device)
            started = time.perf_counter()
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half and args.device != "cpu",
                verbose=False,
            )[0]
            synchronize(args.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            if frame_id >= args.warmup:
                records.append({
                    "frame": frame_id - args.warmup + 1,
                    "model_call_ms": elapsed_ms,
                    "model_call_fps": 1000.0 / elapsed_ms,
                    "preprocess_ms": result.speed["preprocess"],
                    "inference_ms": result.speed["inference"],
                    "postprocess_ms": result.speed["postprocess"],
                })
    finally:
        camera.release()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    latencies = [row["model_call_ms"] for row in records]
    mean_latency = statistics.mean(latencies)
    print(f"Measured frames: {len(records)} after {args.warmup} warm-up frames")
    print(f"Mean model.predict: {mean_latency:.3f} ms")
    print(f"Median model.predict: {statistics.median(latencies):.3f} ms")
    print(f"Mean / minimum rate: {1000.0 / mean_latency:.3f} / {1000.0 / max(latencies):.3f} FPS")
    print(f"Per-frame CSV: {output.resolve()}")


if __name__ == "__main__":
    main()
