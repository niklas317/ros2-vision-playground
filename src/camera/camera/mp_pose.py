#!/usr/bin/env python3
'''
Docstring for camera.camera.mp_pose
'''

from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class MPPose(Node):
    def __init__(self):
        super().__init__('mp_pose')

        
        self.bridge = CvBridge()
        self.frame = None


        # BlazePose (33 landmarks) skeleton connections (same structure as MediaPipe Pose)
        self.POSE_CONNECTIONS = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),

            # Torso
            (11, 12),
            (11, 23), (12, 24),
            (23, 24),

            # Left arm
            (11, 13), (13, 15),
            (15, 17), (15, 19), (15, 21),
            (17, 19),

            # Right arm
            (12, 14), (14, 16),
            (16, 18), (16, 20), (16, 22),
            (18, 20),

            # Left leg
            (23, 25), (25, 27),
            (27, 29), (29, 31),
            (27, 31),

            # Right leg
            (24, 26), (26, 28),
            (28, 30), (30, 32),
            (28, 32),
        ]

        
        
        # Path to Landmark model
        project_root = Path(__file__).resolve().parents[3]
        model_path = project_root / "models" / "pose_landmarker_lite.task"


        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1)
        self.detector = vision.PoseLandmarker.create_from_options(options)


        cv2.namedWindow("POSE", cv2.WINDOW_NORMAL)


        if not model_path.exists():
            self.get_logger().error(f"Pose model not found: {model_path}")
        else:
            self.get_logger().info(f"Using pose model: {model_path}")

        
        self.camera_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        
        self.get_logger().info('Pose Recognition Node started! Listening to /camera/image_raw/')


    def image_callback(self, msg: Image):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            print(1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # timestamp in milliseconds (ROS stamp is in nanoseconds)
            ts_ms = int(msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec / 1e6)

            result = self.detector.detect_for_video(mp_image, ts_ms)

            annotated_rgb = self.draw_landmarks_on_image(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("POSE", annotated_bgr)
            cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f'Pose Pipeline failed! {exc}')
            return
              

    def draw_landmarks_on_image(self, rgb_image: np.ndarray, result) -> np.ndarray:
        annotated = rgb_image.copy()
        h, w = annotated.shape[:2]

        pose_landmarks_list = getattr(result, "pose_landmarks", None)
        if not pose_landmarks_list:
            return annotated

        for pose_landmarks in pose_landmarks_list:
            pts = []
            for lm in pose_landmarks:
                x = int(np.clip(lm.x * w, 0, w - 1))
                y = int(np.clip(lm.y * h, 0, h - 1))
                v = float(getattr(lm, "visibility", 1.0))
                pts.append((x, y, v))

            # Connections
            for a, b in self.POSE_CONNECTIONS:
                xa, ya, va = pts[a]
                xb, yb, vb = pts[b]
                if va < 0.5 or vb < 0.5:
                    continue
                cv2.line(annotated, (xa, ya), (xb, yb), (255, 255, 255), 2)

            # Points
            for x, y, v in pts:
                if v < 0.5:
                    continue
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

        return annotated






def main(args=None):
    rclpy.init(args=args)
    node = MPPose()
    node.get_logger().info('Pose Estimation running')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
