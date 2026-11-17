import os
from glob import glob
from setuptools import setup

package_name = 'custom_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch files (agar launch/ folder hai to install honge, warna kuch nahi hota)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Muzammil',
    maintainer_email='your_email@example.com',
    description='Custom navigation: IA* planner, path smoother and pure pursuit controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = custom_navigation.planner_node:main',
            'smoother_node = custom_navigation.smoother_node:main',
            'controller_node = custom_navigation.controller_node:main',
        ],
    },
)
