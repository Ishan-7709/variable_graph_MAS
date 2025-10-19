#!/usr/bin/env python3
"""Generate simulation video without ROS or display"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
sim_time = 40  # seconds

# Initialize
pos = np.array(spawn, dtype=float).reshape((2, 1))
target = np.array(goal, dtype=float).reshape((2, 1))
trajectory = [pos.flatten().copy()]

# Simulate
steps = int(sim_time / dt)
for _ in range(steps):
    control = leader_gain * (target - pos)
    pos += control * dt
    if _ % 100 == 0:  # Store every 100th step
        trajectory.append(pos.flatten().copy())

trajectory = np.array(trajectory)

# Create video
print(f"Creating video with {len(trajectory)} frames...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Single Agent Navigation')

# Static elements
ax.plot(spawn[0], spawn[1], 'go', markersize=10, label='Spawn')
ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
ax.legend()

# Animated elements
agent_marker = patches.Circle(spawn, 0.2, color='purple')
ax.add_patch(agent_marker)
traj_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)

def update(frame):
    agent_marker.center = (trajectory[frame, 0], trajectory[frame, 1])
    traj_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    return agent_marker, traj_line

anim = animation.FuncAnimation(fig, update, frames=len(trajectory), 
                              interval=33, blit=True, repeat=False)

writer = animation.FFMpegWriter(fps=30, bitrate=1800)
output_path = '/home/rosdevish/ros2_ws/src/variable_graph_MAS/singleagent3d.mp4'
anim.save(output_path, writer=writer)
print(f"Video saved: {output_path}")
plt.close()
