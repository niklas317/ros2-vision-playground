#!/home/liamb/hrs_tutorial_group_b/my_venv/bin/python3

#TODO !/usr/bin/env python3


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
#from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp

from msgs.msg import FaceBoundingBox, FaceBoundingBoxArray

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')

        # QoS similar to your other camera nodes
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()
        self.frame = None

        self.last_header = None

        # MediaPipe face detection setup
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=0,              # 0: short range, 1: full range
            min_detection_confidence=0.5,
        )

        # Subscriber: undistorted image
        self.image_sub = self.create_subscription(Image, '/camera/image_undistorted', self.image_callback, sensor_qos)

        # NEW publisher 1: primary face (single bounding box)
        self.primary_bbox_pub = self.create_publisher(
            FaceBoundingBox,
            '/mediapipe/face_bbox',
            10
        )

        # NEW publisher 2: all faces (array of bboxes)
        self.bbox_array_pub = self.create_publisher(
            FaceBoundingBoxArray,
            '/mediapipe/face_bboxes',
            10
        )

        self.get_logger().info('FaceDetectionNode started, listening to /camera/image_undistorted')

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



    def face_detection(self):

        frame_bgr = self.frame
        ih, iw, _ = frame_bgr.shape

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(frame_rgb)

        boxes_msgs = []

        if not results.detections:
            return frame_bgr, boxes_msgs  # no faces

        for detection in results.detections:
            bbox_rel = detection.location_data.relative_bounding_box

            x_min = int(bbox_rel.xmin * iw)
            y_min = int(bbox_rel.ymin * ih)
            bw    = int(bbox_rel.width * iw)
            bh    = int(bbox_rel.height * ih)
            x_max = x_min + bw
            y_max = y_min + bh

            # clamp to image bounds
            x_min = max(0, min(iw - 1, x_min))
            y_min = max(0, min(ih - 1, y_min))
            x_max = max(0, min(iw - 1, x_max))
            y_max = max(0, min(ih - 1, y_max))

            score = float(detection.score[0]) if detection.score else 0.0

            # draw rectangle
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # fill custom msg
            box_msg = FaceBoundingBox()
            box_msg.x_min = x_min
            box_msg.y_min = y_min
            box_msg.x_max = x_max
            box_msg.y_max = y_max
            box_msg.score = score

            boxes_msgs.append(box_msg)

        return frame_bgr, boxes_msgs
    

    def publish_boxes(self, boxes_msgs):
        """Publish an array of all boxes + the first box as 'primary'."""
        if not boxes_msgs:
            return

        # 1) array of all boxes
        array_msg = FaceBoundingBoxArray()
        if self.last_header is not None:
            array_msg.header = self.last_header
        array_msg.boxes = boxes_msgs
        self.bbox_array_pub.publish(array_msg)

        # 2) primary box (just the first detection)
        primary = boxes_msgs[0]
        self.primary_bbox_pub.publish(primary)
        #if primary > 0.8:
        #    self.primary_bbox_pub.publish(primary)


    def display_loop(self):

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.frame is None:
                continue

            img, boxes_msgs = self.face_detection()
            if img is None:
                continue

            # publish detections
            self.publish_boxes(boxes_msgs)

            # show image with boxes
            cv2.imshow('Face Detection', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break

        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    node.get_logger().info('FaceDetection node started')

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    
    main()
