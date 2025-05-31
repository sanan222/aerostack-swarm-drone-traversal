import launch
import launch_ros.actions
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    scenario_file = launch.actions.DeclareLaunchArgument(
        'scenario_file', default_value='scenario/scenario1.yaml'
    )

    return LaunchDescription([
        scenario_file,
        # Launch the scenario_obstacle_viz node
        Node(
            package='scenario_obstacle_viz',
            executable='viz',
            name='scenario_obstacle_viz',
            parameters=[{"scenario_file": LaunchConfiguration("scenario_file")}]
        )
    ])
