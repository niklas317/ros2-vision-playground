from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package = 'v4l2_camera',
            namespace = 'camera',
            executable = 'v4l2_camera_node',
            name = 'stream',
            parameters = [{
                'video_device': '/dev/video4',
            }],
        ),
        
        Node(
            package = 'camera',
            namespace = 'camera',
            executable = 'usb_camera_sub',
            name = 'camera_sub'
        ),

    ])