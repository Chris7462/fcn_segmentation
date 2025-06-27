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

#   import os
#   from pathlib import Path
#   from typing import List

#   from ament_index_python.packages import get_package_share_directory

#   from launch import LaunchDescription
#   from launch.actions import DeclareLaunchArgument, OpaqueFunction
#   from launch.substitutions import LaunchConfiguration
#   from launch_ros.actions import Node


#   def launch_setup(context) -> List[Node]:
#       """
#       Setup function to create nodes with resolved launch arguments.
#       This allows for dynamic parameter resolution.
#       """
#       # Get launch arguments
#       namespace = LaunchConfiguration('namespace').perform(context)
#       node_name = LaunchConfiguration('node_name').perform(context)

#       # Resolve parameter file path
#       params_file = os.path.join(
#           get_package_share_directory('fcn_segmentation'),
#           'params',
#           'fcn_segmentation.yaml'
#       )

#       # Verify parameter file exists
#       if not Path(params_file).exists():
#           raise FileNotFoundError(f"Parameter file not found: {params_file}")

#       # Create node with dynamic parameters
#       fcn_segmentation_node = Node(
#           package='fcn_segmentation',
#           executable='fcn_segmentation_node',
#           name=node_name,
#           namespace=namespace if namespace else '',
#           output='screen',
#           parameters=[
#               params_file
#           ],
#       )

#       return [fcn_segmentation_node]


#   def generate_launch_description():
#       """Generate launch description with configurable parameters."""

#       # Declare launch arguments for flexibility
#       declared_arguments = [
#           DeclareLaunchArgument(
#               'namespace',
#               default_value='',
#               description='Namespace for the FCN segmentation node'
#           ),
#           DeclareLaunchArgument(
#               'node_name',
#               default_value='fcn_segmentation_node',
#               description='Name of the FCN segmentation node'
#           ),
#       ]

#       return LaunchDescription(
#           declared_arguments + [
#               OpaqueFunction(function=launch_setup)
#           ]
#       )
