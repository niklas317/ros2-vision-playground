#!/usr/bin/env python3
'''
Docstring for src.camera.camera.generate_calibration
'''

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import Image
from pathlib import Path
import glob
import os
import yaml
import time



class CameraCalibration(Node):
    def __init__(self):
        super().__init__('camera_calibration')
        
        self.bridge = CvBridge()

        # self.camera_sub = self.create_subscription(
        #     Image,
        #     'camera/camera_image',
        #     10
        # )

        # Create a Path to the Folder with the Checkerboard screenshots
        ws_root = Path(__file__).resolve().parents[6]
        self.path = ws_root / "Screenshots" / "Calibration_lerobot_cam"

        self.get_logger().info(
            'Camera Calibration Node started!'
        )

        self.calibrate()

        rclpy.shutdown()



    def calibrate(self):
        
        self.get_logger().info(f"Starting calibration using images in: {self.path}")

        # termination criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
              30,
              0.001
            )
        
        # Size of Checkerboard
        chessboard_cols = 10  # x-Richtung
        chessboard_rows = 7   # y-Richtung
       
        # points in world coordinate system (z=0)
        objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)

        square_size = 25.0  # mm 
        objp *= square_size
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        image_paths = glob.glob(os.path.join(self.path, "*.png"))
        if not image_paths:
            self.get_logger().error("No calibration images found!")
            return
        
        gray = None

        for file_path in image_paths:
            #for fname in images:
            image = cv2.imread(file_path)
            if image is None:
                self.get_logger().warn(f"Could not read image: {file_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # Find the chess board corners
            # each box = 25 mm of length
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
        
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
                imgpoints.append(corners2)
        
                cv2.drawChessboardCorners(image, (chessboard_cols, chessboard_rows), corners2, ret)

            if len(objpoints) < 1:
                print(len(objpoints))
                self.get_logger().error("Not enough valid calibration images.")
                return
        
            # calibrate using all images.
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            h,  w = gray.shape[:2]
   
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(objpoints)
        self.get_logger().info(f"Calibration done. Mean reprojection error: {mean_error}")

        # Save as YAML file:
        calib_dict = {
            "image_width": int(w),
            "image_height": int(h),
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "new_camera_matrix": newcameramtx.tolist(),
            "roi": {
                "x": int(roi[0]),
                "y": int(roi[1]),
                "width": int(roi[2]),
                "height": int(roi[3]),
            },
            "mean_reprojection_error": float(mean_error),
        }

        with open("Screenshots/calibration.yaml", "w") as f:
            yaml.safe_dump(calib_dict, f)

        self.get_logger().info("Calibration parameters saved to Screenshots/")


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibration()

    node.destroy_node()
    cv2.destroyAllWindows()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()