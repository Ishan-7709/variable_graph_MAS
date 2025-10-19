#!/usr/bin/env python3
"""Generate 3D flight video with real-time floor mapping"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import yaml

# Load configs
with open('/home/rosdevish/ros2_ws/src/variable_graph_MAS/src/easySingle.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('/home/rosdevish/ros2_ws/src/variable_graph_MAS/src/floor_objects.yaml', 'r') as f:
    floor_data = yaml.safe_load(f)

# Parameters
spawn = config['Agents']['NamesandPos'][0][1]
goal = config['Agents']['Leaders']['Zoe']['Target']
leader_gain = config['pnpParameters']['leaderGain']
dt = 0.01
sim_time = 25

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
steps = int(sim_time / dt)
for step in range(steps):
    # Navigation
    nav = target - pos
    nav[2] = 0
    control = leader_gain * nav
    pos += control * dt
    pos[2] = constant_height
    
    if step % 50 == 0:
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
fig = plt.figure(figsize=(16, 7))

# 3D plot
ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_xlim(0, 10)
ax3d.set_ylim(0, 10)
ax3d.set_zlim(0, 6)
ax3d.set_xlabel('X (m)')
ax3d.set_ylabel('Y (m)')
ax3d.set_zlabel('Z (m)')
ax3d.set_title('3D Agent Flight')

# Static elements
ax3d.scatter([spawn[0]], [spawn[1]], [spawn[2]], c='green', marker='o', s=100, label='Spawn')
ax3d.scatter([goal[0]], [goal[1]], [goal[2]], c='red', marker='*', s=400, label='Goal')
ax3d.legend()

# Animated elements (3D)
agent_scatter = ax3d.scatter([], [], [], c='purple', marker='o', s=200)
traj_line, = ax3d.plot([], [], [], 'b-', alpha=0.6, linewidth=2)

# 2D map
ax2d = fig.add_subplot(122)
ax2d.set_xlim(0, map_size)
ax2d.set_ylim(0, map_size)
ax2d.set_xlabel('X (m)')
ax2d.set_ylabel('Y (m)')
ax2d.set_title('Real-Time Sonar Floor Map')
ax2d.set_aspect('equal')

# Plot floor objects as ground truth
for obj in floor_objects:
    if obj['type'] == 'circle':
        circle = plt.Circle(obj['position'], obj['radius'], fill=False, color='white', linestyle='--', alpha=0.3)
        ax2d.add_patch(circle)
    elif obj['type'] == 'rectangle':
        rect = plt.Rectangle(
            (obj['position'][0] - obj['size'][0]/2, obj['position'][1] - obj['size'][1]/2),
            obj['size'][0], obj['size'][1], fill=False, color='white', linestyle='--', alpha=0.3
        )
        ax2d.add_patch(rect)

# Map image
colors = ['black', 'blue', 'cyan', 'yellow', 'red']
cmap = LinearSegmentedColormap.from_list('sonar', colors, N=100)
map_im = ax2d.imshow(np.zeros((grid_size, grid_size)), cmap=cmap, origin='lower',
                      extent=[0, map_size, 0, map_size], alpha=0.8, vmin=0, vmax=1)

# Agent position marker on map
agent_marker_2d, = ax2d.plot([], [], 'mo', markersize=10, label='Agent')
ax2d.legend()

def update(frame):
    # Update 3D plot
    agent_scatter._offsets3d = ([trajectory[frame, 0]], [trajectory[frame, 1]], [trajectory[frame, 2]])
    traj_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    traj_line.set_3d_properties(trajectory[:frame+1, 2])
    ax3d.view_init(elev=20, azim=frame*0.5)
    
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
                    current_map[grid_y, grid_x] += intensity * 0.02
    
    map_normalized = np.clip(current_map / (np.max(current_map) + 1e-6), 0, 1)
    map_im.set_data(map_normalized)
    
    # Update agent position on map
    agent_marker_2d.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
    
    return agent_scatter, traj_line, map_im, agent_marker_2d

anim = animation.FuncAnimation(fig, update, frames=len(trajectory),
                              interval=50, blit=False, repeat=False)

writer = animation.FFMpegWriter(fps=20, bitrate=3000)
output_path = '/home/rosdevish/ros2_ws/src/variable_graph_MAS/singleagent3d.mp4'
anim.save(output_path, writer=writer)
print(f"Video with mapping saved: {output_path}")
plt.close()
