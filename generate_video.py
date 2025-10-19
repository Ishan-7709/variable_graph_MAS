#!/usr/bin/env python3
"""Generate 3D simulation video without ROS or display"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import yaml

# Load config
with open('/home/rosdevish/ros2_ws/src/variable_graph_MAS/src/easySingle.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Parameters
spawn = config['Agents']['NamesandPos'][0][1]
goal = config['Agents']['Leaders']['Zoe']['Target']
leader_gain = config['pnpParameters']['leaderGain']
dt = 0.01
sim_time = 25  # seconds

# Initialize (3D)
pos = np.array(spawn, dtype=float).reshape((3, 1))
target = np.array(goal, dtype=float).reshape((3, 1))
constant_height = float(spawn[2])
trajectory = [pos.flatten().copy()]

# Simulate
steps = int(sim_time / dt)
for step in range(steps):
    # XY navigation only
    nav = target - pos
    nav[2] = 0  # No Z control
    control = leader_gain * nav
    
    pos += control * dt
    pos[2] = constant_height  # Force constant height
    
    if step % 50 == 0:  # Store every 50th step
        trajectory.append(pos.flatten().copy())

trajectory = np.array(trajectory)

# Create 3D video
print(f"Creating 3D video with {len(trajectory)} frames...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 6)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Agent Navigation - Constant Height Flight')

# Static elements
ax.scatter([spawn[0]], [spawn[1]], [spawn[2]], c='green', marker='o', s=100, label='Spawn')
ax.scatter([goal[0]], [goal[1]], [goal[2]], c='red', marker='*', s=400, label='Goal')
ax.legend()

# Animated elements
agent_scatter = ax.scatter([], [], [], c='purple', marker='o', s=200)
traj_line, = ax.plot([], [], [], 'b-', alpha=0.6, linewidth=2)

def update(frame):
    # Update agent position
    agent_scatter._offsets3d = ([trajectory[frame, 0]], 
                                 [trajectory[frame, 1]], 
                                 [trajectory[frame, 2]])
    # Update trajectory
    traj_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    traj_line.set_3d_properties(trajectory[:frame+1, 2])
    
    # Rotate view for better 3D effect
    ax.view_init(elev=20, azim=frame*0.5)
    
    return agent_scatter, traj_line

anim = animation.FuncAnimation(fig, update, frames=len(trajectory), 
                              interval=50, blit=False, repeat=False)

writer = animation.FFMpegWriter(fps=20, bitrate=2000)
output_path = '/home/rosdevish/ros2_ws/src/variable_graph_MAS/singleagent3d.mp4'
anim.save(output_path, writer=writer)
print(f"3D video saved: {output_path}")
plt.close()
