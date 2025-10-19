#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
import os


class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')
        
        # Declare parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('agent_name', 'Zoe')
        
        # Get parameters
        config_file = self.get_parameter('config_file').value
        self.agent_name = self.get_parameter('agent_name').value
        
        # Load configuration
        self.load_config(config_file)
        
        # Agent data storage
        self.agent_pos = None
        self.trajectory = []
        
        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.agent_name}/odom',
            self.odom_callback,
            10
        )
        
        # Setup matplotlib
        self.setup_plot()
        
        # Timer for updating plot
        self.viz_timer = self.create_timer(0.033, self.update_plot)  # 30Hz
        
        self.get_logger().info(f'Visualization node started for agent {self.agent_name}')
    
    def load_config(self, config_file):
        """Load YAML configuration file"""
        if not os.path.exists(config_file):
            self.get_logger().error(f'Config file not found: {config_file}')
            raise FileNotFoundError(f'Config file not found: {config_file}')
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract workspace bounds
        self.wksp_upper = self.config['Wksp_bnds']['upper'][0][0]
        self.wksp_lower = self.config['Wksp_bnds']['lower'][0][0]
        
        # Extract goal
        self.goal = self.config['Agents']['Leaders'][self.agent_name]['Target']
    
    def setup_plot(self):
        """Initialize matplotlib figure"""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Set limits
        self.ax.set_xlim(self.wksp_lower, self.wksp_upper)
        self.ax.set_ylim(self.wksp_lower, self.wksp_upper)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Single Agent Navigation')
        
        # Plot goal marker
        self.ax.plot(self.goal[0], self.goal[1], 'r*', markersize=20, label='Goal')
        
        # Initialize agent marker (will be updated)
        self.agent_marker = patches.Circle((0, 0), 0.2, color='purple', animated=True)
        self.ax.add_patch(self.agent_marker)
        
        # Initialize trajectory line
        self.traj_line, = self.ax.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Trajectory')
        
        self.ax.legend()
        plt.show(block=False)
    
    def odom_callback(self, msg):
        """Callback for odometry messages"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.agent_pos = np.array([x, y])
        
        # Store trajectory (subsample to avoid memory issues)
        if len(self.trajectory) == 0 or len(self.trajectory) % 10 == 0:
            self.trajectory.append([x, y])
    
    def update_plot(self):
        """Update matplotlib visualization"""
        if self.agent_pos is None:
            return
        
        try:
            # Update agent position
            self.agent_marker.center = (self.agent_pos[0], self.agent_pos[1])
            
            # Update trajectory
            if len(self.trajectory) > 1:
                traj_array = np.array(self.trajectory)
                self.traj_line.set_data(traj_array[:, 0], traj_array[:, 1])
            
            # Redraw
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().warn(f'Plot update error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
