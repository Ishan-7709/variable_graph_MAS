#!/usr/bin/env python3
"""Simple 2D floor mapping from sonar detections"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        
        # Declare parameters
        self.declare_parameter('agent_name', 'Zoe')
        self.declare_parameter('map_resolution', 0.1)  # meters per cell
        self.declare_parameter('map_size', 10.0)  # workspace size
        
        # Get parameters
        self.agent_name = self.get_parameter('agent_name').value
        resolution = self.get_parameter('map_resolution').value
        map_size = self.get_parameter('map_size').value
        
        # Initialize map
        self.grid_size = int(map_size / resolution)
        self.resolution = resolution
        self.map_size = map_size
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))
        
        # Subscribe to sonar scans
        self.sonar_sub = self.create_subscription(
            PointCloud2,
            f'/{self.agent_name}/sonar_scan',
            self.sonar_callback,
            10
        )
        
        # Timer to save map periodically
        self.save_timer = self.create_timer(2.0, self.save_map)
        
        self.get_logger().info(f'Mapping node initialized: {self.grid_size}x{self.grid_size} grid')
    
    def sonar_callback(self, msg):
        """Process sonar detections and update map"""
        # Parse PointCloud2
        for point in pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True):
            x, y, z, intensity = point
            
            # Convert world coordinates to grid coordinates
            grid_x = int((x / self.map_size) * self.grid_size)
            grid_y = int((y / self.map_size) * self.grid_size)
            
            # Check bounds
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                # Accumulate intensity
                self.occupancy_grid[grid_y, grid_x] += intensity
    
    def save_map(self):
        """Save current map as image"""
        if np.sum(self.occupancy_grid) == 0:
            return  # No detections yet
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Normalize and plot
        map_normalized = np.clip(self.occupancy_grid / (np.max(self.occupancy_grid) + 1e-6), 0, 1)
        
        # Custom colormap: black (empty) to yellow (high intensity)
        colors = ['black', 'blue', 'cyan', 'yellow', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('sonar', colors, N=n_bins)
        
        im = ax.imshow(map_normalized, cmap=cmap, origin='lower', extent=[0, self.map_size, 0, self.map_size])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Sonar Floor Map')
        plt.colorbar(im, ax=ax, label='Detection Intensity')
        
        # Save
        output_path = '/home/rosdevish/ros2_ws/src/variable_graph_MAS/maps/floor_map.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f'Map saved: {output_path}')


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Saving final map...')
        node.save_map()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
