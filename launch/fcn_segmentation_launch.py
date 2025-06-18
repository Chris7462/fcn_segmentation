from os.path import join

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = join(
        get_package_share_directory('fcn_segmentation'), 'params',
        'fcn_segmentation.yaml'
    )

    fcn_segmentation_node = Node(
        package='fcn_segmentation',
        executable='fcn_segmentation_node',
        name='fcn_segmentation_node',
        output='screen',
        parameters=[params]
    )

    return LaunchDescription([
        fcn_segmentation_node
    ])
