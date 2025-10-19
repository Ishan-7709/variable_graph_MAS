#!/usr/bin/env python3
"""Generate 3D flight video with real-time floor mapping"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import yaml

# Load configs
with open('/home/rosdevish/ros2_ws/src/variable_graph_MAS/src/easySingle.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('/home/rosdevish/ros2_ws/src/variable_graph_MAS/src/floor_objects.yaml', 'r') as f:
    floor_data = yaml.safe_load(f)

# Parameters - SLOWER SPEED
spawn = config['Agents']['NamesandPos'][0][1]
goal = config['Agents']['Leaders']['Zoe']['Target']
leader_gain = 0.15  # Much slower than 0.5
dt = 0.01
sim_time = 60  # Longer simulation

# Initialize agent (3D)
pos = np.array(spawn, dtype=float).reshape((3, 1))
target = np.array(goal, dtype=float).reshape((3, 1))
constant_height = float(spawn[2])
trajectory = [pos.flatten().copy()]

# Initialize map
map_size = 10.0
map_resolution = 0.1
grid_size = int(map_size / map_resolution)
occupancy_grid = np.zeros((grid_size, grid_size))

# Sonar parameters
sonar_altitude = floor_data['sonar_config']['altitude']
beam_angle = floor_data['sonar_config']['beam_angle']
footprint_radius = sonar_altitude * np.tan(np.radians(beam_angle / 2))

# Floor objects
floor_objects = floor_data['floor_objects']
object_height = 0.75  # meters

def draw_cylinder_3d(ax, center, radius, height, color='gray', alpha=0.6):
    """Draw a cylinder (tire) in 3D"""
    z = np.linspace(0, height, 20)
    theta = np.linspace(0, 2 * np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)

def draw_box_3d(ax, center, size, height, color='gray', alpha=0.6):
    """Draw a rectangular box in 3D"""
    w, h = size[0], size[1]
    x0, y0 = center[0] - w/2, center[1] - h/2
    x1, y1 = center[0] + w/2, center[1] + h/2
    
    vertices = [
        [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],  # Bottom
        [x0, y0, height], [x1, y0, height], [x1, y1, height], [x0, y1, height]  # Top
    ]
    
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[2], vertices[3]]   # Bottom
    ]
    
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_collection3d(poly)

def draw_line_3d(ax, points, width, height, color='gray', alpha=0.6):
    """Draw a line/cable in 3D"""
    p0, p1 = np.array(points[0]), np.array(points[1])
    
    # Draw as a series of small cylinders
    num_segments = 20
    for i in range(num_segments):
        t = i / num_segments
        center = p0 + t * (p1 - p0)
        
        # Create small cylinder segment
        z = np.linspace(0, height, 5)
        theta = np.linspace(0, 2 * np.pi, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = center[0] + width * np.cos(theta_grid)
        y_grid = center[1] + width * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)

def check_detection(agent_xy, obj):
    """Check if object is in sonar footprint"""
    if obj['type'] == 'circle':
        pos = np.array(obj['position'])
        dist = np.linalg.norm(pos - agent_xy)
        return dist <= (footprint_radius + obj['radius']), pos, obj['intensity']
    elif obj['type'] == 'rectangle':
        pos = np.array(obj['position'])
        dist = np.linalg.norm(pos - agent_xy)
        return dist <= footprint_radius, pos, obj['intensity']
    elif obj['type'] == 'line':
        points = np.array(obj['points'])
        for point in points:
            dist = np.linalg.norm(point - agent_xy)
            if dist <= footprint_radius:
                return True, point, obj['intensity']
    return False, None, 0

# Simulate
print("Simulating agent flight...")
steps = int(sim_time / dt)
for step in range(steps):
    # Navigation
    nav = target - pos
    nav[2] = 0
    control = leader_gain * nav
    pos += control * dt
    pos[2] = constant_height
    
    if step % 100 == 0:  # Sample every 100 steps for smoother video
        trajectory.append(pos.flatten().copy())
        
        # Sonar scan
        agent_xy = pos[:2, 0]
        for obj in floor_objects:
            detected, det_pos, intensity = check_detection(agent_xy, obj)
            if detected and det_pos is not None:
                # Update map
                grid_x = int((det_pos[0] / map_size) * grid_size)
                grid_y = int((det_pos[1] / map_size) * grid_size)
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    occupancy_grid[grid_y, grid_x] += intensity * 0.5

trajectory = np.array(trajectory)

# Create figure with 3D flight and 2D map side-by-side
print(f"Creating video with {len(trajectory)} frames...")
fig = plt.figure(figsize=(18, 8))

# 3D plot
ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_xlim(0, 10)
ax3d.set_ylim(0, 10)
ax3d.set_zlim(0, 6)
ax3d.set_xlabel('X (m)')
ax3d.set_ylabel('Y (m)')
ax3d.set_zlabel('Z (m)')
ax3d.set_title('3D Agent Flight with Floor Objects')

# Draw floor objects in 3D
print("Drawing floor objects...")
for obj in floor_objects:
    if obj['type'] == 'circle':
        draw_cylinder_3d(ax3d, obj['position'], obj['radius'], object_height, color='orange')
    elif obj['type'] == 'rectangle':
        draw_box_3d(ax3d, obj['position'], obj['size'], object_height, color='brown')
    elif obj['type'] == 'line':
        draw_line_3d(ax3d, obj['points'], obj['width'], object_height, color='gray')

# Static elements
ax3d.scatter([spawn[0]], [spawn[1]], [spawn[2]], c='green', marker='o', s=150, label='Spawn', zorder=10)
ax3d.scatter([goal[0]], [goal[1]], [goal[2]], c='red', marker='*', s=500, label='Goal', zorder=10)
ax3d.legend(loc='upper left')

# Animated elements (3D)
agent_scatter = ax3d.scatter([], [], [], c='purple', marker='o', s=250, zorder=10)
traj_line, = ax3d.plot([], [], [], 'b-', alpha=0.7, linewidth=2.5, zorder=5)

# 2D map
ax2d = fig.add_subplot(122)
ax2d.set_xlim(0, map_size)
ax2d.set_ylim(0, map_size)
ax2d.set_xlabel('X (m)')
ax2d.set_ylabel('Y (m)')
ax2d.set_title('Real-Time Sonar Floor Map')
ax2d.set_aspect('equal')
ax2d.set_facecolor('#1a1a1a')

# Plot floor objects as ground truth on map
for obj in floor_objects:
    if obj['type'] == 'circle':
        circle = plt.Circle(obj['position'], obj['radius'], fill=False, 
                          color='white', linestyle='--', alpha=0.4, linewidth=2)
        ax2d.add_patch(circle)
        ax2d.text(obj['position'][0], obj['position'][1], obj['name'], 
                 color='white', fontsize=8, ha='center', alpha=0.5)
    elif obj['type'] == 'rectangle':
        rect = plt.Rectangle(
            (obj['position'][0] - obj['size'][0]/2, obj['position'][1] - obj['size'][1]/2),
            obj['size'][0], obj['size'][1], fill=False, 
            color='white', linestyle='--', alpha=0.4, linewidth=2
        )
        ax2d.add_patch(rect)
        ax2d.text(obj['position'][0], obj['position'][1], obj['name'], 
                 color='white', fontsize=8, ha='center', alpha=0.5)
    elif obj['type'] == 'line':
        points = np.array(obj['points'])
        ax2d.plot(points[:, 0], points[:, 1], 'w--', alpha=0.4, linewidth=2)

# Map image
colors = ['#000000', '#001a33', '#004d99', '#00ccff', '#ffff00', '#ff6600']
cmap = LinearSegmentedColormap.from_list('sonar', colors, N=100)
map_im = ax2d.imshow(np.zeros((grid_size, grid_size)), cmap=cmap, origin='lower',
                      extent=[0, map_size, 0, map_size], alpha=0.9, vmin=0, vmax=1)

# Agent position marker and footprint on map
footprint_circle = plt.Circle((0, 0), footprint_radius, fill=False, 
                              color='magenta', linestyle='-', linewidth=2, alpha=0.6)
ax2d.add_patch(footprint_circle)
agent_marker_2d, = ax2d.plot([], [], 'mo', markersize=12, label='Agent', markeredgewidth=2, markeredgecolor='white')
ax2d.legend(loc='upper right')

def update(frame):
    # Update 3D plot
    agent_scatter._offsets3d = ([trajectory[frame, 0]], [trajectory[frame, 1]], [trajectory[frame, 2]])
    traj_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    traj_line.set_3d_properties(trajectory[:frame+1, 2])
    
    # Slow rotation for better view
    ax3d.view_init(elev=25, azim=30 + frame*0.3)
    
    # Update map (accumulate detections up to current frame)
    current_map = np.zeros((grid_size, grid_size))
    for f in range(frame + 1):
        agent_xy = trajectory[f, :2]
        for obj in floor_objects:
            detected, det_pos, intensity = check_detection(agent_xy, obj)
            if detected and det_pos is not None:
                grid_x = int((det_pos[0] / map_size) * grid_size)
                grid_y = int((det_pos[1] / map_size) * grid_size)
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    current_map[grid_y, grid_x] += intensity * 0.03
    
    map_normalized = np.clip(current_map / (np.max(current_map) + 1e-6), 0, 1)
    map_im.set_data(map_normalized)
    
    # Update agent position and footprint on map
    agent_marker_2d.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
    footprint_circle.center = (trajectory[frame, 0], trajectory[frame, 1])
    
    return agent_scatter, traj_line, map_im, agent_marker_2d, footprint_circle

print("Rendering animation...")
anim = animation.FuncAnimation(fig, update, frames=len(trajectory),
                              interval=100, blit=False, repeat=False)  # 100ms = 10fps

writer = animation.FFMpegWriter(fps=10, bitrate=4000)
output_path = '/home/rosdevish/ros2_ws/src/variable_graph_MAS/singleagent3d.mp4'
anim.save(output_path, writer=writer)
print(f"Video saved: {output_path} ({len(trajectory)/10:.1f} seconds)")
plt.close()
