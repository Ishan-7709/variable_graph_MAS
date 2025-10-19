#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Vector3
from std_msgs.msg import Header
import numpy as np
import yaml
import os


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        
        # Declare parameters
        self.declare_parameter('agent_name', 'Zoe')
        self.declare_parameter('config_file', '')
        
        # Get parameters
        self.agent_name = self.get_parameter('agent_name').value
        config_file = self.get_parameter('config_file').value
        
        # Load configuration
        self.load_config(config_file)
        
        # Agent state
        self.pos = np.array(self.initial_pos, dtype=float).reshape((2, 1))
        self.target = np.array(self.goal, dtype=float).reshape((2, 1))
        
        # Control parameters
        self.dt = 0.01  # 100Hz control rate
        self.leader_gain = self.config['pnpParameters']['leaderGain']
        
        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry, 
            f'/{self.agent_name}/odom', 
            10
        )
        
        # Timer for control loop
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info(f'Agent {self.agent_name} initialized at {self.pos.flatten()}')
        self.get_logger().info(f'Target: {self.target.flatten()}')
    
    def load_config(self, config_file):
        """Load YAML configuration file"""
        if not os.path.exists(config_file):
            self.get_logger().error(f'Config file not found: {config_file}')
            raise FileNotFoundError(f'Config file not found: {config_file}')
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract agent-specific config
        agent_data = None
        for name_pos in self.config['Agents']['NamesandPos']:
            if name_pos[0] == self.agent_name:
                agent_data = name_pos
                break
        
        if agent_data is None:
            self.get_logger().error(f'Agent {self.agent_name} not found in config')
            raise ValueError(f'Agent {self.agent_name} not found in config')
        
        self.initial_pos = agent_data[1]
        self.goal = self.config['Agents']['Leaders'][self.agent_name]['Target']
    
    def simple_navigation(self, current, goal):
        """Simple radial navigation function (goal - current)"""
        return goal - current
    
    def control_loop(self):
        """Main control loop running at 100Hz"""
        # Compute control input (simple navigation to goal for now)
        control_input = self.leader_gain * self.simple_navigation(self.pos, self.target)
        
        # Euler integration
        self.pos += control_input * self.dt
        
        # Publish odometry
        self.publish_odometry()
        
        # Check if goal reached
        distance_to_goal = np.linalg.norm(self.target - self.pos)
        if distance_to_goal < 0.1:
            self.get_logger().info(f'Goal reached! Distance: {distance_to_goal:.4f}')
    
    def publish_odometry(self):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header = Header()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'world'
        odom_msg.child_frame_id = self.agent_name
        
        # Position
        odom_msg.pose.pose.position = Point(
            x=float(self.pos[0, 0]),
            y=float(self.pos[1, 0]),
            z=0.0
        )
        
        # Orientation (identity quaternion for 2D)
        odom_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # Velocity (not computing for now)
        odom_msg.twist.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        odom_msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
        
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
