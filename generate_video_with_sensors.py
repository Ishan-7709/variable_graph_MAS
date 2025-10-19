#!/usr/bin/env python3
"""Generate video: 3D flight (left) + Sonar/Mag (right stack) - VERY SLOW"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import yaml

# Load configs
with open('/home/rosdevish/ros2_ws/src/variable_graph_MAS/src/easySingle.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('/home/rosdevish/ros2_ws/src/variable_graph_MAS/src/floor_objects.yaml', 'r') as f:
    floor_data = yaml.safe_load(f)

# Parameters - VERY SLOW
spawn = config['Agents']['NamesandPos'][0][1]
goal = config['Agents']['Leaders']['Zoe']['Target']
leader_gain = 0.08  # Even slower - was 0.15
dt = 0.01
sim_time = 120  # 2 minutes simulation

# Initialize agent
pos = np.array(spawn, dtype=float).reshape((3, 1))
target = np.array(goal, dtype=float).reshape((3, 1))
constant_height = float(spawn[2])
trajectory = [pos.flatten().copy()]

# Initialize maps
map_size = 10.0
map_resolution = 0.1
grid_size = int(map_size / map_resolution)
sonar_map = np.zeros((grid_size, grid_size))
mag_map = np.ones((grid_size, grid_size)) * 48000

# Sonar parameters
sonar_altitude = floor_data['sonar_config']['altitude']
beam_angle = floor_data['sonar_config']['beam_angle']
footprint_radius = sonar_altitude * np.tan(np.radians(beam_angle / 2))

# Floor objects
floor_objects = floor_data['floor_objects']
object_height = 0.75

# Magnetic signatures
object_mag_strength = {
    'tire_1': 500,
    'box_1': 250,
    'cable_1': 80,
    'target_1': 800
}

def draw_cylinder_3d(ax, center, radius, height, color='gray', alpha=0.6):
    z = np.linspace(0, height, 20)
    theta = np.linspace(0, 2 * np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)

def draw_box_3d(ax, center, size, height, color='gray', alpha=0.6):
    w, h = size[0], size[1]
    x0, y0 = center[0] - w/2, center[1] - h/2
    x1, y1 = center[0] + w/2, center[1] + h/2
    vertices = [
        [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
        [x0, y0, height], [x1, y0, height], [x1, y1, height], [x0, y1, height]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[2], vertices[3]]
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_collection3d(poly)

def draw_line_3d(ax, points, width, height, color='gray', alpha=0.6):
    p0, p1 = np.array(points[0]), np.array(points[1])
    num_segments = 20
    for i in range(num_segments):
        t = i / num_segments
        center = p0 + t * (p1 - p0)
        z = np.linspace(0, height, 5)
        theta = np.linspace(0, 2 * np.pi, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = center[0] + width * np.cos(theta_grid)
        y_grid = center[1] + width * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)

def check_detection(agent_xy, obj):
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

def mag_dipole_field(sample_pos, obj_pos, strength):
    dist = np.linalg.norm(sample_pos - obj_pos)
    if dist < 0.1:
        dist = 0.1
    return strength / (dist**2 + 1.0)

# Simulate
print("Simulating SLOW flight with dual sensors...")
steps = int(sim_time / dt)
sonar_snapshots = [sonar_map.copy()]
mag_snapshots = [mag_map.copy()]

for step in range(steps):
    nav = target - pos
    nav[2] = 0
    control = leader_gain * nav
    pos += control * dt
    pos[2] = constant_height
    
    if step % 100 == 0:
        trajectory.append(pos.flatten().copy())
        agent_xy = pos[:2, 0]
        
        # Sonar
        for obj in floor_objects:
            detected, det_pos, intensity = check_detection(agent_xy, obj)
            if detected and det_pos is not None:
                grid_x = int((det_pos[0] / map_size) * grid_size)
                grid_y = int((det_pos[1] / map_size) * grid_size)
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    sonar_map[grid_y, grid_x] += intensity * 0.02
        
        # Magnetometer
        for obj in floor_objects:
            obj_name = obj['name']
            if obj_name in object_mag_strength and 'position' in obj:
                obj_pos = np.array(obj['position'])
                mag_strength = object_mag_strength[obj_name]
                
                for dx in range(-15, 16):
                    for dy in range(-15, 16):
                        grid_x = int((agent_xy[0] / map_size) * grid_size) + dx
                        grid_y = int((agent_xy[1] / map_size) * grid_size) + dy
                        
                        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                            world_x = (grid_x / grid_size) * map_size
                            world_y = (grid_y / grid_size) * map_size
                            sample_pos = np.array([world_x, world_y])
                            
                            field_anomaly = mag_dipole_field(sample_pos, obj_pos, mag_strength)
                            mag_map[grid_y, grid_x] += field_anomaly * 0.08
        
        sonar_snapshots.append(sonar_map.copy())
        mag_snapshots.append(mag_map.copy())

trajectory = np.array(trajectory)

# Create figure: Left = 3D, Right = Sonar (top) + Mag (bottom)
print(f"Creating video with {len(trajectory)} frames...")
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])

# LEFT: 3D (spans both rows)
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_xlim(0, 10)
ax3d.set_ylim(0, 10)
ax3d.set_zlim(0, 6)
ax3d.set_xlabel('X (m)', fontsize=11)
ax3d.set_ylabel('Y (m)', fontsize=11)
ax3d.set_zlabel('Z (m)', fontsize=11)
ax3d.set_title('ROS Agent Over Seabed', fontsize=15, fontweight='bold', pad=20)

# Draw objects
print("Drawing floor objects...")
for obj in floor_objects:
    if obj['type'] == 'circle':
        draw_cylinder_3d(ax3d, obj['position'], obj['radius'], object_height, color='orange')
    elif obj['type'] == 'rectangle':
        draw_box_3d(ax3d, obj['position'], obj['size'], object_height, color='brown')
    elif obj['type'] == 'line':
        draw_line_3d(ax3d, obj['points'], obj['width'], object_height, color='gray')

ax3d.scatter([spawn[0]], [spawn[1]], [spawn[2]], c='green', marker='o', s=200, label='Spawn', zorder=10)
ax3d.scatter([goal[0]], [goal[1]], [goal[2]], c='red', marker='*', s=600, label='Goal', zorder=10)
ax3d.legend(loc='upper left', fontsize=11)

agent_scatter = ax3d.scatter([], [], [], c='purple', marker='o', s=300, zorder=10)
traj_line, = ax3d.plot([], [], [], 'b-', alpha=0.7, linewidth=3, zorder=5)

# TOP RIGHT: Sonar
ax_sonar = fig.add_subplot(gs[0, 1])
ax_sonar.set_xlim(0, map_size)
ax_sonar.set_ylim(0, map_size)
ax_sonar.set_xlabel('X (m)', fontsize=10)
ax_sonar.set_ylabel('Y (m)', fontsize=10)
ax_sonar.set_title('Processed Sonar Data', fontsize=13, fontweight='bold')
ax_sonar.set_aspect('equal')
ax_sonar.set_facecolor('#0a0a0a')

sonar_colors = ['#000000', '#001a33', '#004d99', '#00ccff', '#ffff00', '#ff6600']
sonar_cmap = LinearSegmentedColormap.from_list('sonar', sonar_colors, N=100)
sonar_im = ax_sonar.imshow(np.zeros((grid_size, grid_size)), cmap=sonar_cmap, origin='lower',
                            extent=[0, map_size, 0, map_size], alpha=0.95, vmin=0, vmax=1)
plt.colorbar(sonar_im, ax=ax_sonar, label='Intensity', fraction=0.046, pad=0.04)

# BOTTOM RIGHT: Magnetometer
ax_mag = fig.add_subplot(gs[1, 1])
ax_mag.set_xlim(0, map_size)
ax_mag.set_ylim(0, map_size)
ax_mag.set_xlabel('X (m)', fontsize=10)
ax_mag.set_ylabel('Y (m)', fontsize=10)
ax_mag.set_title('Processed Mag Data', fontsize=13, fontweight='bold')
ax_mag.set_aspect('equal')
ax_mag.set_facecolor('#f0f5fa')

mag_im = ax_mag.imshow(np.ones((grid_size, grid_size)) * 48000, cmap='coolwarm', origin='lower',
                        extent=[0, map_size, 0, map_size], vmin=47700, vmax=48300)
plt.colorbar(mag_im, ax=ax_mag, label='Field (nT)', fraction=0.046, pad=0.04)

plt.tight_layout()

def update(frame):
    # 3D
    agent_scatter._offsets3d = ([trajectory[frame, 0]], [trajectory[frame, 1]], [trajectory[frame, 2]])
    traj_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    traj_line.set_3d_properties(trajectory[:frame+1, 2])
    ax3d.view_init(elev=25, azim=30 + frame*0.2)
    
    # Sonar
    sonar_normalized = np.clip(sonar_snapshots[frame] / (np.max(sonar_snapshots[frame]) + 1e-6), 0, 1)
    sonar_im.set_data(sonar_normalized)
    
    # Mag
    mag_im.set_data(mag_snapshots[frame])
    
    return agent_scatter, traj_line, sonar_im, mag_im

print("Rendering animation...")
anim = animation.FuncAnimation(fig, update, frames=len(trajectory),
                              interval=100, blit=False, repeat=False)

writer = animation.FFMpegWriter(fps=10, bitrate=6000)
output_path = '/home/rosdevish/ros2_ws/src/variable_graph_MAS/singleagent_dual_sensors.mp4'
anim.save(output_path, writer=writer)
print(f"Video saved: {output_path} ({len(trajectory)/10:.1f} seconds)")
print("Layout: 3D view (left) | Sonar (top-right) + Mag (bottom-right)")
plt.close()
