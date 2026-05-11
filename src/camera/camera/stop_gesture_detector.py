#!/usr/bin/env python3
"""Deterministic stop-gesture detector for compressed camera images."""

from __future__ import annotations

from math import atan2, degrees

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class StopGestureDetector(Node):
    """Detect a stop gesture using only deterministic image processing."""

    def __init__(self):
        super().__init__('stop_gesture_detector')

        self.declare_parameter('compressed_topic', '/camera/image_raw/compressed')
        self.declare_parameter('show_debug_views', True)
        self.declare_parameter('min_contour_area', 4000.0)

        self.compressed_topic = self.get_parameter('compressed_topic').value
        self.show_debug_views = bool(self.get_parameter('show_debug_views').value)
        self.min_contour_area = float(self.get_parameter('min_contour_area').value)

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage,
            self.compressed_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.subscription

        if self.show_debug_views:
            cv2.namedWindow('hand_raw', cv2.WINDOW_NORMAL)
            cv2.namedWindow('hand_annotated', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f'Listening on {self.compressed_topic} for stop gestures'
        )

    def image_callback(self, msg: CompressedImage):
        """Process one compressed frame and annotate the result."""

        try:
            frame_bgr = self._decode_compressed_image(msg)
            if frame_bgr is None:
                self.get_logger().warn('Could not decode compressed image')
                return

            mask = self._build_skin_mask(frame_bgr)
            contour = self._largest_contour(mask)

            annotated = frame_bgr.copy()
            stop_detected = False

            if contour is not None:
                stop_detected, diagnostics = self._classify_stop(contour)
                self._draw_annotations(annotated, contour, diagnostics, stop_detected)
            else:
                cv2.putText(
                    annotated,
                    'No hand found',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            if self.show_debug_views:
                try:
                    cv2.imshow('hand_raw', frame_bgr)
                    cv2.imshow('hand_annotated', annotated)
                    cv2.waitKey(1)
                except cv2.error as exc:
                    self.get_logger().warn(f'cv2.imshow failed: {exc}')

            if stop_detected:
                self.get_logger().info('STOP gesture detected')

        except Exception as exc:
            self.get_logger().error(f'Gesture pipeline failed: {exc}')

    def _decode_compressed_image(self, msg: CompressedImage):
        """Decode a ROS compressed image message into BGR format."""

        try:
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            buffer = np.frombuffer(msg.data, dtype=np.uint8)
            if buffer.size == 0:
                return None
            return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _build_skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Build a deterministic skin-like mask from HSV and YCrCb thresholds."""

        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

        hsv_mask = cv2.inRange(hsv, (0, 25, 40), (25, 220, 255))
        ycrcb_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _largest_contour(self, mask: np.ndarray):
        """Return the largest external contour in the mask."""

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0]
        if cv2.contourArea(contour) < self.min_contour_area:
            return None
        return contour

    def _classify_stop(self, contour):
        """Classify a contour as a stop gesture using geometric rules."""

        area = float(cv2.contourArea(contour))
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        hull_points = cv2.convexHull(contour, returnPoints=True)
        hull_area = float(cv2.contourArea(hull_points)) if len(hull_points) else 0.0
        solidity = area / hull_area if hull_area > 0.0 else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        centroid = self._contour_centroid(contour)

        defects = self._count_valid_defects(contour, hull_indices, h)
        finger_tips = self._extract_finger_tips(hull_points, y, h, w)

        primary_fingers = finger_tips[:4]
        finger_count = len(primary_fingers)
        finger_line_angle = self._finger_line_angle(primary_fingers)
        palm_visible = area > 0.10 * (w * h) and solidity > 0.45
        tips_above_palm = bool(primary_fingers) and max(pt[1] for pt in primary_fingers) < centroid[1]
        angle_ok = abs(finger_line_angle) <= 30.0 if finger_count >= 2 else False
        spread_ok = self._finger_spread_ok(primary_fingers, w, h)

        stop_detected = bool(
            palm_visible
            and tips_above_palm
            and finger_count >= 4
            and defects >= 3
            and angle_ok
            and spread_ok
        )

        diagnostics = {
            'area': area,
            'hull_area': hull_area,
            'solidity': solidity,
            'bbox': (x, y, w, h),
            'centroid': centroid,
            'defects': defects,
            'finger_tips': primary_fingers,
            'finger_line_angle': finger_line_angle,
            'palm_visible': palm_visible,
            'tips_above_palm': tips_above_palm,
            'angle_ok': angle_ok,
            'spread_ok': spread_ok,
        }
        return stop_detected, diagnostics

    def _count_valid_defects(self, contour, hull_indices, bbox_height: int) -> int:
        """Count convexity defects that look like finger gaps."""

        if hull_indices is None or len(hull_indices) < 3:
            return 0

        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None:
            return 0

        valid_defects = 0
        min_depth = max(8.0, 0.02 * bbox_height)

        for defect in defects:
            s, e, f, depth = defect[0]
            start = contour[s][0]
            end = contour[e][0]
            far = contour[f][0]

            a = np.linalg.norm(end - start)
            b = np.linalg.norm(far - start)
            c = np.linalg.norm(end - far)
            if b <= 0.0 or c <= 0.0:
                continue

            cosine = np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)
            angle = degrees(np.arccos(cosine))
            depth_px = depth / 256.0

            if angle < 95.0 and depth_px >= min_depth:
                valid_defects += 1

        return valid_defects

    def _extract_finger_tips(self, hull_points, bbox_y: int, bbox_h: int, bbox_w: int):
        """Extract fingertip candidates from the upper hull of the contour."""

        if hull_points is None or len(hull_points) == 0:
            return []

        points = [tuple(pt[0]) for pt in hull_points]
        top_limit = bbox_y + int(0.55 * bbox_h)
        top_points = [pt for pt in points if pt[1] <= top_limit]
        if not top_points:
            return []

        top_points.sort(key=lambda pt: pt[0])
        min_gap = max(10, int(0.10 * bbox_w))

        clusters: list[list[tuple[int, int]]] = []
        for pt in top_points:
            if not clusters:
                clusters.append([pt])
                continue

            if abs(pt[0] - clusters[-1][-1][0]) <= min_gap:
                clusters[-1].append(pt)
            else:
                clusters.append([pt])

        fingertips = [min(cluster, key=lambda p: p[1]) for cluster in clusters]
        fingertips.sort(key=lambda pt: pt[1])
        return fingertips

    def _finger_line_angle(self, finger_tips):
        """Estimate the tilt of the finger line in degrees."""

        if len(finger_tips) < 2:
            return 0.0

        ordered = sorted(finger_tips, key=lambda pt: pt[0])
        left = ordered[0]
        right = ordered[-1]
        return degrees(atan2(right[1] - left[1], right[0] - left[0]))

    def _finger_spread_ok(self, finger_tips, bbox_w: int, bbox_h: int) -> bool:
        """Ensure fingertip candidates are spread like four separate fingers."""

        if len(finger_tips) < 4:
            return False

        xs = sorted(pt[0] for pt in finger_tips)
        min_gap = max(12, int(0.05 * bbox_w))
        if any((xs[idx + 1] - xs[idx]) < min_gap for idx in range(len(xs) - 1)):
            return False

        ys = [pt[1] for pt in finger_tips]
        return (max(ys) - min(ys)) <= int(0.35 * bbox_h)

    def _contour_centroid(self, contour):
        """Return the centroid of a contour."""

        moments = cv2.moments(contour)
        if moments['m00'] == 0.0:
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w / 2.0, y + h / 2.0)

        return (
            moments['m10'] / moments['m00'],
            moments['m01'] / moments['m00'],
        )

    def _draw_annotations(self, frame, contour, diagnostics, stop_detected: bool):
        """Draw contour, hull, bounding box and status overlays."""

        x, y, w, h = diagnostics['bbox']
        hull = cv2.convexHull(contour)
        centroid = diagnostics['centroid']

        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)

        for tip in diagnostics['finger_tips']:
            cv2.circle(frame, tip, 8, (255, 255, 0), -1)

        status = 'STOP' if stop_detected else 'HAND' if diagnostics['palm_visible'] else 'NO STOP'
        color = (0, 0, 255) if stop_detected else (0, 255, 255)
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
        )

        lines = [
            f'solidity={diagnostics["solidity"]:.2f} defects={diagnostics["defects"]}',
            f'finger_tips={len(diagnostics["finger_tips"])} angle={diagnostics["finger_line_angle"]:.1f}deg',
            f'angle_ok={diagnostics["angle_ok"]} spread_ok={diagnostics["spread_ok"]}',
        ]
        base_y = 75
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, base_y + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )


def main(args=None):
    """Entry point for the stop gesture detector node."""

    rclpy.init(args=args)
    node = StopGestureDetector()

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
