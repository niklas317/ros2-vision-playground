#!/usr/bin/env python3
'''
Docstring for src.camera.camera.generate_calibration
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import Image

from ultralytics import YOLO

class YoloV11(Node):
    def __init__(self):
        super().__init__('yolov11')

        self.bridge = CvBridge()

        # Params (so you can override from CLI)
        self.declare_parameter('model', 'models/yolov8n.pt')   # or a path to your .pt
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('device', 'cpu')         # 'cpu' or '0' for GPU

        self.model_name = self.get_parameter('model').value
        self.conf = float(self.get_parameter('conf').value)
        self.device = str(self.get_parameter('device').value)

        # Load YOLO once
        self.get_logger().info(f"Loading YOLO model: {self.model_name} (device={self.device}, conf={self.conf})")
        self.model = YOLO(self.model_name)

        # OpenCV window
        cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)

        # Camera Subscriber
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_cb,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            'YOLOV11 node started, listening on /image_raw'
        )


    
    def camera_cb(self, msg: Image):
        try:
            # Convert ROS Image -> OpenCV (BGR)
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # YOLO inference
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf,
                device=self.device,
                verbose=False
            )

            # Draw detections
            annotated = results[0].plot()

            # Display
            cv2.imshow("YOLO", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Pressed q -> shutting down.")
                rclpy.shutdown()

        except Exception as exc:
            self.get_logger().error(f'Image conversion/display error: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloV11()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()