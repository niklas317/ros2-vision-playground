#!/usr/bin/env python3
'''
Node for subscribing to a camera with USB connection

Publish the USB Camera:
  ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video4

Requires:
  sudo apt install python3-numpy python3-opencv
'''

import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image


class UsbCameraSubscriber(Node):
    def __init__(self):
        super().__init__('usb_camera_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'image_raw',          # topic from v4l2_camera
            self.image_callback,
            10
        )

        self.subscription
        self.get_logger().info(
            'USB camera subscriber node started, listening on /image_raw'
        )

    def image_callback(self, msg: Image):
        try:
            # Convert raw bytes to numpy array and reshape
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )

            # v4l2_camera usually publishes rgb8
            if msg.encoding == 'rgb8':
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = img  # might already be BGR, etc.

            cv2.imshow('USB Camera', frame_bgr)
            cv2.waitKey(1)  # important: lets OpenCV process GUI events

        except Exception as exc:
            self.get_logger().error(f'Image conversion/display error: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraSubscriber()

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
