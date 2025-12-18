#!/home/liamb/hrs_tutorial_group_b/my_venv/bin/python3
# TODO: /usr/bin/env python3

import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from msgs.msg import HandGesture, HandGestureArray

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GestureRecognizerNode(Node):
    def __init__(self):
        super().__init__('gesture_recognizer_node')

        # QoS wie bei den anderen Vision-Nodes
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()
        self.frame = None
        self.last_header = None

        # === Modelpfad bestimmen (relativ zum Workspace-Root) ===
        project_root = Path(__file__).resolve().parents[3]
        model_path = project_root / "models" / "gesture_recognizer.task"

        if not model_path.exists():
            self.get_logger().error(f"Gesture model not found: {model_path}")
        else:
            self.get_logger().info(f"Using gesture model: {model_path}")

        base_options = python.BaseOptions(model_asset_path=str(model_path))

        self.options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,  # max. 2 Hände
        )

        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(
            self.options
        )

        self.timestamp_ms = 0

        # === Subscriber & Publisher ===
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_undistorted',
            self.image_callback,
            sensor_qos
        )

        self.gesture_pub = self.create_publisher(
            HandGestureArray,
            '/mediapipe/gesture_prediction',
            10
        )

        self.get_logger().info(
            'GestureRecognizerNode started: '
            'subscribing to /camera/image_undistorted, '
            'publishing /mediapipe/gesture_prediction'
        )

    # -------- ROS Callbacks --------
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


    def run_gesture_recognizer(self):
        frame_bgr = self.frame
        h, w, _ = frame_bgr.shape

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # timestamp for VIDEO-Mode
        self.timestamp_ms += 33  # ~30 FPS

        result = self.gesture_recognizer.recognize_for_video(
            mp_image, self.timestamp_ms
        )

        if not result.gestures:
            return frame_bgr, None  # No hands detected

        gestures_msgs = []

        # result.gestures, result.handedness und result.hand_landmarks
        for hand_idx, gesture_list in enumerate(result.gestures):
            if len(gesture_list) == 0:
                continue

            top_gesture = gesture_list[0]
            gesture_name = top_gesture.category_name
            score = float(top_gesture.score)

            # Handedness (left/right)
            handed = "Unknown"
            if result.handedness and len(result.handedness) > hand_idx:
                if len(result.handedness[hand_idx]) > 0:
                    handed = result.handedness[hand_idx][0].category_name

            # build message
            hg = HandGesture()
            if self.last_header is not None:
                hg.header = self.last_header
            else:
                hg.header.stamp = self.get_clock().now().to_msg()
                hg.header.frame_id = "camera_link"

            hg.gesture = gesture_name
            hg.score = score
            hg.handedness = handed

            gestures_msgs.append(hg)

            # Landmarksfor debugging plotted in the image
            if result.hand_landmarks and len(result.hand_landmarks) > hand_idx:
                landmarks = result.hand_landmarks[hand_idx]
                for lm in landmarks:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame_bgr, (cx, cy), 3, (0, 0, 255), -1)

                # Text with gesture output
                text = f"{handed}: {gesture_name} ({score:.2f})"
                cv2.putText(frame_bgr, text, (10, 30 + 30 * hand_idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(gestures_msgs) == 0:
            return frame_bgr, None

        array_msg = HandGestureArray()
        if self.last_header is not None:
            array_msg.header = self.last_header
        else:
            array_msg.header.stamp = self.get_clock().now().to_msg()
            array_msg.header.frame_id = "camera_link"

        array_msg.gestures = gestures_msgs

        return frame_bgr, array_msg

    def display_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.frame is None:
                continue

            img, gesture_array_msg = self.run_gesture_recognizer()
            if img is None:
                continue

            if gesture_array_msg is not None:
                self.gesture_pub.publish(gesture_array_msg)

            # Für Screenshot (Task 5)
            cv2.imshow('Gesture Recognition', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = GestureRecognizerNode()
    node.get_logger().info('GestureRecognizerNode running')

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
