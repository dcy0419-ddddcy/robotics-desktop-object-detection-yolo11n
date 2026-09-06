import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ResultSubscriber(Node):
    def __init__(self) -> None:
        super().__init__("desktop_object_result_subscriber")
        self.subscription = self.create_subscription(
            String, "/object_detection/result", self.receive, 10
        )

    def receive(self, message: String) -> None:
        result = json.loads(message.data)
        objects = ", ".join(
            f"{item['class_name']}({item['confidence']:.2f})"
            for item in result["detections"]
        ) or "none"
        self.get_logger().info(
            f"frame={result['frame_id']} fps={result['fps']:.2f} objects={objects}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ResultSubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
