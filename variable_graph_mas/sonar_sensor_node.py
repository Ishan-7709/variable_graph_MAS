#!/usr/bin/env python3
"""Simulated downward-looking sonar sensor"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import yaml
import os
import struct


class SonarSensorNode(Node):
    def __init__(self):
        super().__init__('sonar_sensor_node')
        
        # Declare parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('floor_objects_file', '')
        self.declare_parameter('agent_name', 'Zoe')
        
        # Get parameters
        config_file = self.get_parameter('config_file').value
        floor_file = self.get_parameter('floor_objects_file').value
        self.agent_name = self.get_parameter('agent_name').value
        
        # Load configuration
        self.load_config(config_file, floor_file)
        
        # Agent position
        self.agent_pos = None
        
        # Subscriber to agent odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.agent_name}/odom',
            self.odom_callback,
            10
        )
        
        # Publisher for sonar detections
        self.sonar_pub = self.create_publisher(
            PointCloud2,
            f'/{self.agent_name}/sonar_scan',
            10
        )
        
        # Timer for sonar scanning
        scan_rate = self.sonar_config['scan_rate']
        self.scan_timer = self.create_timer(1.0 / scan_rate, self.scan_callback)
        
        self.get_logger().info(f'Sonar sensor initialized for {self.agent_name}')
        self.get_logger().info(f'Loaded {len(self.floor_objects)} floor objects')
    
    def load_config(self, config_file, floor_file):
        """Load agent and floor object configurations"""
        # Load agent config
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load floor objects
        if not os.path.exists(floor_file):
            self.get_logger().warn(f'Floor objects file not found: {floor_file}')
            self.floor_objects = []
            self.sonar_config = {
                'altitude': 3.0,
                'beam_angle': 45,
                'resolution': 0.05,
                'scan_rate': 10
            }
            return
        
        with open(floor_file, 'r') as f:
            floor_data = yaml.safe_load(f)
        
        self.floor_objects = floor_data.get('floor_objects', [])
        self.sonar_config = floor_data.get('sonar_config', {})
    
    def odom_callback(self, msg):
        """Update agent position from odometry"""
        self.agent_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
    
    def check_object_in_footprint(self, obj, agent_xy):
        """Check if object is within sonar footprint"""
        # Calculate footprint radius at floor
        altitude = self.sonar_config['altitude']
        beam_half_angle = np.radians(self.sonar_config['beam_angle'] / 2)
        footprint_radius = altitude * np.tan(beam_half_angle)
        
        obj_type = obj['type']
        
        if obj_type == 'circle':
            # Check if circle center is in footprint
            pos = np.array(obj['position'])
            dist = np.linalg.norm(pos - agent_xy)
            return dist <= (footprint_radius + obj['radius'])
        
        elif obj_type == 'rectangle':
            # Check if rectangle center is in footprint
            pos = np.array(obj['position'])
            dist = np.linalg.norm(pos - agent_xy)
            size = np.array(obj['size'])
            return dist <= (footprint_radius + np.linalg.norm(size)/2)
        
        elif obj_type == 'line':
            # Check if any point on line is in footprint
            points = np.array(obj['points'])
            for point in points:
                dist = np.linalg.norm(point - agent_xy)
                if dist <= footprint_radius:
                    return True
            return False
        
        return False
    
    def scan_callback(self):
        """Simulate sonar scan"""
        if self.agent_pos is None:
            return
        
        agent_xy = self.agent_pos[:2]
        detections = []
        
        # Check each floor object
        for obj in self.floor_objects:
            if self.check_object_in_footprint(obj, agent_xy):
                # Object detected - add detection point
                if obj['type'] == 'circle':
                    pos = obj['position']
                    intensity = obj['intensity']
                    detections.append([pos[0], pos[1], 0.0, intensity])
                
                elif obj['type'] == 'rectangle':
                    pos = obj['position']
                    intensity = obj['intensity']
                    detections.append([pos[0], pos[1], 0.0, intensity])
                
                elif obj['type'] == 'line':
                    # Sample points along line
                    points = np.array(obj['points'])
                    num_samples = 10
                    for i in range(num_samples):
                        t = i / (num_samples - 1)
                        point = points[0] + t * (points[1] - points[0])
                        detections.append([point[0], point[1], 0.0, obj['intensity']])
        
        # Publish detections as PointCloud2
        if len(detections) > 0:
            self.publish_point_cloud(detections)
    
    def publish_point_cloud(self, detections):
        """Publish detections as PointCloud2 message"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'world'
        
        # Define fields: x, y, z, intensity
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Pack point data
        point_data = []
        for det in detections:
            point_data.append(struct.pack('ffff', *det))
        
        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(detections)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16  # 4 floats * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = b''.join(point_data)
        
        self.sonar_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SonarSensorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
