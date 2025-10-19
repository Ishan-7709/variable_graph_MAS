#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('variable_graph_mas')
    
    # Default config file path
    default_config = os.path.join(pkg_dir, 'config', 'easySingle.yaml')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to YAML configuration file'
    )
    
    agent_name_arg = DeclareLaunchArgument(
        'agent_name',
        default_value='Zoe',
        description='Name of the agent'
    )
    
    # Get launch configurations
    config_file = LaunchConfiguration('config_file')
    agent_name = LaunchConfiguration('agent_name')
    
    # Agent node
    agent_node = Node(
        package='variable_graph_mas',
        executable='agent_node',
        name='agent_node',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'agent_name': agent_name
        }]
    )
    
    # Visualization node
    viz_node = Node(
        package='variable_graph_mas',
        executable='viz_node',
        name='viz_node',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'agent_name': agent_name
        }]
    )
    
    return LaunchDescription([
        config_file_arg,
        agent_name_arg,
        agent_node,
        viz_node
    ])
