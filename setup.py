from setuptools import setup
import os
from glob import glob

package_name = 'variable_graph_mas'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('src/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ishan Agrawal',
    maintainer_email='ishan.agrawal@integer-tech.com',
    description='Multi-agent system with variable graph topology using PnP control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent_node = variable_graph_mas.agent_node:main',
            'viz_node = variable_graph_mas.viz_node:main',
        ],
    },
)
