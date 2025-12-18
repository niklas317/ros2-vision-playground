#!/home/liamb/hrs_tutorial_group_b/my_venv/bin/python3
# TODO: /usr/bin/env python3

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from msgs.msg import PoseLandmarks

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # === QoS wie bei deinen anderen Nodes ===
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()
        self.frame = None
        self.last_header = None

        # === MediaPipe PoseLandmarker ===
        # Projekt-Root:  .../src/face_detection/face_detection/pose_estimation_node.py
        # → drei Ebenen hoch = Workspace-Root
        project_root = Path(__file__).resolve().parents[3]
        model_path = project_root / "models" / "pose_landmarker_lite.task"

        if not model_path.exists():
            self.get_logger().error(f"Pose model not found: {model_path}")
        else:
            self.get_logger().info(f"Using pose model: {model_path}")

        base_options = python.BaseOptions(model_asset_path=str(model_path))

        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            self.options
        )

        self.timestamp_ms = 0

        # === ROS Subscriber / Publisher ===
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_undistorted',
            self.image_callback,
            sensor_qos
        )

        self.pose_pub = self.create_publisher(
            PoseLandmarks,
            '/mediapipe/pose_landmarkers',
            10
        )

        self.get_logger().info(
            'PoseEstimationNode started: '
            'subscribing to /camera/image_undistorted, '
            'publishing /mediapipe/pose_landmarkers'
        )

    def image_callback(self, msg: Image):
        try:
            
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if frame_bgr is None:
                self.get_logger().warn('Bridge command unsuccessful!')
                return

            self.frame = frame_bgr
            self.last_header = msg.header

        except Exception as exc:
            self.get_logger().error(f'CvBridge conversion failed! {exc}')
            return
        

    def run_pose_landmarker(self):


        frame_bgr = self.frame
        h, w, _ = frame_bgr.shape

        # Mediapipe is using RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        self.timestamp_ms += 33  # ~30 FPS

        result = self.pose_landmarker.detect_for_video(
            mp_image, self.timestamp_ms
        )

        if not result.pose_landmarks:
            return frame_bgr, None  # No person detected

        # Only chose first pose (Pose 0)
        pose_landmarks = result.pose_landmarks[0]

        xs = []
        ys = []
        zs = []
        vis = []

        for lm in pose_landmarks:
            xs.append(lm.x)
            ys.append(lm.y)
            zs.append(lm.z)
            vis.append(lm.visibility)

            # Für Debug: draw points into image
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)

        # ROS-Message
        msg = PoseLandmarks()
        if self.last_header is not None:
            msg.header = self.last_header
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"

        msg.x = xs
        msg.y = ys
        msg.z = zs
        msg.visibility = vis

        return frame_bgr, msg

    def display_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.frame is None:
                continue

            img, pose_msg = self.run_pose_landmarker()
            if img is None:
                continue

            # publish detections
            if pose_msg is not None:
                self.pose_pub.publish(pose_msg)

            # For Screenshot show pose in the image
            cv2.imshow('Pose Landmarks', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    node.get_logger().info('PoseEstimationNode running')

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
