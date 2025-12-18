from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='niklas',
    maintainer_email='niklasderpeter@gmx.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'usb_camera_sub = camera.usb_camera_sub:main',
            'calibrate = camera.generate_calibration:main',
            'yolo = camera.yolov11:main',
            'mp_pose = camera.mp_pose:main',
            'mp_upperbody_pose = camera.mp_upperbody_pose:main',
            'testprint = camera.testprint:main',
        ],
    },
)
