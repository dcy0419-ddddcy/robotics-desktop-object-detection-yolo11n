import json
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("desktop_object_detector")
        self.declare_parameter("model", "models/best.pt")
        self.declare_parameter("source", "/dev/video0")
        self.declare_parameter("device", "0")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("half", True)
        self.declare_parameter("display", True)

        self.model = YOLO(self.get_parameter("model").value)
        source = str(self.get_parameter("source").value)
        self.camera = cv2.VideoCapture(int(source) if source.isdigit() else source)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

        self.publisher = self.create_publisher(String, "/object_detection/result", 10)
        self.frame_id = 0
        self.smoothed_fps = 0.0
        self.timer = self.create_timer(0.001, self.process_frame)

    def process_frame(self) -> None:
        success, frame = self.camera.read()
        if not success:
            self.get_logger().error("Camera frame acquisition failed")
            return

        device = str(self.get_parameter("device").value)
        started = time.perf_counter()
        result = self.model.predict(
            frame,
            imgsz=int(self.get_parameter("imgsz").value),
            conf=float(self.get_parameter("conf").value),
            iou=float(self.get_parameter("iou").value),
            device=device,
            half=bool(self.get_parameter("half").value) and device != "cpu",
            verbose=False,
        )[0]
        current_fps = 1.0 / max(time.perf_counter() - started, 1e-6)
        self.smoothed_fps = current_fps if self.smoothed_fps == 0.0 else (
            0.9 * self.smoothed_fps + 0.1 * current_fps
        )

        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(value, 1) for value in box.xyxy[0].tolist()],
            })

        message = String()
        message.data = json.dumps({
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
            "frame_id": self.frame_id,
            "fps": round(self.smoothed_fps, 2),
            "detections": detections,
        }, separators=(",", ":"))
        self.publisher.publish(message)

        if bool(self.get_parameter("display").value):
            display = result.plot()
            cv2.putText(display, f"FPS: {self.smoothed_fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("ROS 2 Desktop Object Detection - press Q to exit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rclpy.shutdown()

        self.frame_id += 1

    def close(self) -> None:
        self.camera.release()
        cv2.destroyAllWindows()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
