#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import yaml
import os


class VizNode3D(Node):
    def __init__(self):
        super().__init__('viz_node_3d')
        
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
        self.frames = []  # Store frames for MP4
        self.spawn_pos = None  # Initial spawn position
        
        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.agent_name}/odom',
            self.odom_callback,
            10
        )
        
        # Setup matplotlib 3D
        self.setup_plot()
        
        # Timer for updating plot
        self.viz_timer = self.create_timer(0.033, self.update_plot)  # 30Hz
        
        self.get_logger().info(f'3D Visualization node started for agent {self.agent_name}')
    
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
        
        # Extract goal and spawn position
        self.goal = self.config['Agents']['Leaders'][self.agent_name]['Target']
        for name_pos in self.config['Agents']['NamesandPos']:
            if name_pos[0] == self.agent_name:
                self.spawn_pos = name_pos[1]
                break
    
    def setup_plot(self):
        """Initialize 3D matplotlib figure"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set limits
        self.ax.set_xlim(self.wksp_lower, self.wksp_upper)
        self.ax.set_ylim(self.wksp_lower, self.wksp_upper)
        self.ax.set_zlim(0, 6)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('3D Agent Navigation - Constant Height Flight')
        
        # Plot goal marker
        self.ax.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]], 
                       c='red', marker='*', s=400, label='Goal')
        
        # Plot spawn position as small sphere
        if self.spawn_pos:
            self.ax.scatter([self.spawn_pos[0]], [self.spawn_pos[1]], [self.spawn_pos[2]], 
                           c='green', marker='o', s=100, label='Spawn')
        
        # Initialize agent marker (will be updated)
        self.agent_scatter = self.ax.scatter([], [], [], c='purple', marker='o', s=200)
        
        # Initialize trajectory line
        self.traj_line, = self.ax.plot([], [], [], 'b-', alpha=0.6, linewidth=2, label='Trajectory')
        
        self.ax.legend()
        plt.show(block=False)
    
    def odom_callback(self, msg):
        """Callback for odometry messages"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.agent_pos = np.array([x, y, z])
        
        # Store trajectory (subsample to avoid memory issues)
        if len(self.trajectory) == 0 or len(self.trajectory) % 5 == 0:
            self.trajectory.append([x, y, z])
    
    def update_plot(self):
        """Update 3D matplotlib visualization"""
        if self.agent_pos is None:
            return
        
        try:
            # Update agent position
            self.agent_scatter._offsets3d = ([self.agent_pos[0]], 
                                             [self.agent_pos[1]], 
                                             [self.agent_pos[2]])
            
            # Update trajectory
            if len(self.trajectory) > 1:
                traj_array = np.array(self.trajectory)
                self.traj_line.set_data(traj_array[:, 0], traj_array[:, 1])
                self.traj_line.set_3d_properties(traj_array[:, 2])
            
            # Redraw
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # Capture frame for video (every 3rd frame to reduce file size)
            if len(self.frames) % 3 == 0:
                self.fig.canvas.draw()
                frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                self.frames.append(frame)
            
        except Exception as e:
            self.get_logger().warn(f'Plot update error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = VizNode3D()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down and saving video...')
    finally:
        # Save video
        if len(node.frames) > 0:
            try:
                node.get_logger().info(f'Saving {len(node.frames)} frames to singleagent3d.mp4')
                
                # Create animation from frames
                fig_save = plt.figure(figsize=(10, 8))
                ax_save = fig_save.add_subplot(111)
                im = ax_save.imshow(node.frames[0])
                ax_save.axis('off')
                
                def update_frame(i):
                    im.set_array(node.frames[i])
                    return [im]
                
                anim = animation.FuncAnimation(fig_save, update_frame, frames=len(node.frames), 
                                              interval=33, blit=True)
                
                writer = animation.FFMpegWriter(fps=30, bitrate=1800)
                anim.save('singleagent3d.mp4', writer=writer)
                node.get_logger().info('Video saved as singleagent3d.mp4')
                plt.close(fig_save)
            except Exception as e:
                node.get_logger().error(f'Failed to save video: {e}')
        
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
