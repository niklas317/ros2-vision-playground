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
from cv_bridge import CvBridge


class UsbCameraSubscriber(Node):
    def __init__(self):
        super().__init__('usb_camera_subscriber')

        self.bridge = CvBridge()
        
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

        self.camera_pub = self.create_publisher(Image, 'camera_image', 10)


    
            
    def image_callback(self, msg: Image):
        try:
            # Convert ROS Image -> OpenCV (BGR)
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            cv2.imshow('USB Camera', frame_bgr)
            cv2.waitKey(1)

            # Republish OpenCV -> ROS Image
            out_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
            out_msg.header = msg.header
            self.camera_pub.publish(out_msg)

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
