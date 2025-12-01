from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('model_predictive_speed_and_steer_control_cpp')
    params_file = os.path.join(pkg_share, 'config', 'demo_params.yaml')

    return LaunchDescription(
        [
            Node(
                package='model_predictive_speed_and_steer_control_cpp',
                executable='model_predictive_speed_and_steer_control',
                name='model_predictive_speed_and_steer_control',
                output='screen',
                parameters=[params_file],
            )
        ]
    )
