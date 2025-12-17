#!/usr/bin/env python3
"""
TUM - ICS AiNex CameraSubscriberCompressed Demo for ROS 2 Jazzy
----------------------------------------
Subscribes to JPEG-compressed images and raw images on /camera_image/compressed and /camera_image,
shows frames with OpenCV, and displays CameraInfo.

Requires:
  sudo apt install python3-numpy python3-opencv

Msgs:
    sensor_msgs/CompressedImage
    sensor_msgs/CameraInfo
"""
from typing import Tuple
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Point, Vector3

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.cb_group = ReentrantCallbackGroup()

        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe compressed images
        self.sub_compressed = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',
            self.image_callback_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.sub_compressed

        # Subscribe camera info
        self.sub_camerainfo = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.cb_group
        )
        self.sub_camerainfo

        # State variables
        self.camera_info_received = False

        self.frame = None # BGR Frame
        self.frame_gs = None
        self.frame_binary = None

        self.red_binary_frame = None
        self.green_binary_frame = None
        self.blue_binary_frame = None

        self.detected = None
        self.publisher_com_ = self.create_publisher(msg_type=Point, topic="center_of_mass", qos_profile=10)
        self.publisher_circle_ = self.create_publisher(msg_type=Vector3, topic="circle", qos_profile=10)

        self.center_point: tuple = None
        self.circle: tuple = None

    def publish_msg_com(self):
        msg = Point()
        msg.x, msg.y = self.center_point
        msg.x = float(msg.x)
        msg.y = float(msg.y)
        self.publisher_com_.publish(msg=msg)

    def publish_msg_circle(self):
        msg = Vector3()
        msg.x, msg.y, msg.z = self.circle
        msg.x = float(msg.x)
        msg.y = float(msg.y)
        msg.z = float(msg.z)
        self.publisher_circle_.publish(msg=msg)


    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.get_logger().info(
                f'Camera Info received: {msg.width}x{msg.height}\n'
                f'K: {msg.k}\n'
                f'D: {msg.d}'
            )
            print(f'Camera Info received: {msg.width}x{msg.height}')
            print(f'Intrinsic matrix K: {msg.k}')
            print(f'Distortion coeffs D: {msg.d}')
            self.camera_info_received = True

    def image_callback_compressed(self, msg: CompressedImage):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                self.get_logger().warn('JPEG decode returned None')
                return

            self.frame = frame
            self.frame_gs = self.bgr_to_grayscale(frame)
            self.frame_binary = self.frame_to_binary(self.frame_gs, 127, 255)
            
            self.red_binary_frame = self.color_extraction(frame, [0,170,90], [15,255,255]) + self.color_extraction(frame, [165,170,90], [179,255,255])
            self.green_binary_frame = self.color_extraction(frame, [40,127,90], [80,255,255])
            self.blue_binary_frame = self.color_extraction(frame, [90,50,90], [130,255,255])           
            
            try: 
                biggest_spot = self.find_biggest_spot(self.green_binary_frame)
                center_coordinates = self.center_of_spot(biggest_spot)
                self.detected = cv2.cvtColor(biggest_spot, cv2.COLOR_GRAY2BGR)
                self.draw_circle_in_frame(self.detected, center_coordinates, radius=2, color=(0,255,0))
                self.draw_circle_in_frame(self.detected, center_coordinates, radius=10, color=(0,0,255))
            except:
                pass
            

        except Exception as exc:
            self.get_logger().error(f'Decode error in compressed image: {exc}')

    def process_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        if key == ord('c'):
            self.show_compressed = True
            self.get_logger().info('Switched to compressed image')
        return True

    def display_loop(self):
        while rclpy.ok():
            if self.frame is not None:
                # Display the compressed image
                cv2.imshow('Camera Subscrber', self.frame)
                #cv2.imshow('Greyscale Image', self.frame_gs)
                #cv2.imshow('Binary Image', self.frame_binary)

                #cv2.imshow('Color Extraction Red', self.red_binary_frame)
                #cv2.imshow('Color Extraction Green', self.green_binary_frame)
                #cv2.imshow('Color Extraction Blue', self.blue_binary_frame)
                #cv2.imshow('Circular shapes', self.circle_detector())
                
                #try:
                    #cv2.imshow('Blob extraction', self.detected)
                #except:
                #    pass

            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

    def bgr_to_grayscale(self, bgr_frame) -> None:
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    def frame_to_binary(self, bgr_frame, threshold: int = 127, max_val: int = 255):
        _, frame_binary = cv2.threshold(bgr_frame, threshold, max_val, cv2.THRESH_BINARY)
        return frame_binary

    def color_extraction(self, bgr_frame, lower_threshold_mask: Tuple[int, int ,int], upper_threshold_mask: Tuple[int, int ,int]):
        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV) # Convert from BSG to HSV (Hue, Saturation, Value)
        color_binary_mask = cv2.inRange(hsv_frame, np.array(lower_threshold_mask), np.array(upper_threshold_mask)) # Creates mask based on thresholds
        _, color_binary_frame = cv2.threshold(color_binary_mask, 0, 255, cv2.THRESH_BINARY) # Turns pixelvals binary with x < lower => 0, lower < x < upper => 1
        return color_binary_frame
        
    def find_biggest_spot(self, binary_frame):
        
        initial_erosion = cv2.erode(binary_frame, np.ones((3,3), np.uint8), iterations = 2)
        dilation = cv2.dilate(initial_erosion, np.ones((3,3), np.uint8), iterations = 6)
        post_erosion = cv2.erode(dilation, np.ones((3,3), np.uint8), iterations = 2)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(post_erosion, connectivity=8, ltype=cv2.CV_32S)

        biggest_spot = 0
        largest_spot_label = 1 

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > biggest_spot:
                biggest_spot = area
                largest_spot_label = i

        biggest_spot_mask = np.zeros_like(post_erosion, dtype=np.uint8)
        biggest_spot_mask[labels == largest_spot_label] = 255  

        return biggest_spot_mask

    def center_of_spot(self, masked_framed):
        if np.sum(masked_framed) == 0:
            return (-1,-1)
        y_coordinates, x_coordinates = np.where(masked_framed > 0)
        center = (int(np.average(x_coordinates)), int(np.average(y_coordinates)))
        self.center_point = center
        self.publish_msg_com()

        return center
    
    def circle_detector(self):
        gs = cv2.equalizeHist(self.frame_gs)
        gs = cv2.GaussianBlur(gs, (7,7), 3.0)
        circles = cv2.HoughCircles(
            gs, cv2.HOUGH_GRADIENT, 
            dp=1.1, 
            minDist=50, 
            param1 = 180, 
            param2 = 45, 
            minRadius = 8, 
            maxRadius = 80 
            )

        view = cv2.cvtColor(self.frame_gs, cv2.COLOR_GRAY2BGR)
        if circles is None:
            return view
        
        circles = np.round(circles[0, :]).astype(int)
        h, w = self.frame_gs.shape
        cx, cy = w//2, h//2
        x,y,r = min(circles, key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)

        cv2.circle(view, (x,y), r, (0, 255, 0), 2)
        cv2.circle(view, (x,y), 3, (0, 0 ,255), -1)
        self.circle = (x, y, r)
        self.publish_msg_circle()
        
        return view

    def draw_circle_in_frame(self, frame, center_coordinates, radius=10, color=(0,0,255), thickness=2):
        cv2.circle(frame, center_coordinates, radius, color, thickness)

    
def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    node.get_logger().info('CameraSubscriber node started')

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
