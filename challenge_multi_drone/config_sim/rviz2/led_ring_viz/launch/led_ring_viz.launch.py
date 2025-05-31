from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def get_viz_nodes(context):
    ns_list = LaunchConfiguration('namespace_list').perform(context).split(',')
    return [
        Node(
            package='led_ring_visualizer',
            executable='led_ring_visualizer',
            name='led_ring_visualizer',
            output='screen',
            parameters=[
                {"namespace_list": ns_list}
            ]
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
                'namespace_list',
                description='List of drone namespaces separated by commas. E.g: "drone0, drone1".',
            ),
        OpaqueFunction(function=get_viz_nodes),
    ])