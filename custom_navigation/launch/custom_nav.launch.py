"""Starts the three custom navigation nodes together.

    ros2 launch custom_navigation custom_nav.launch.py use_sim_time:=true
    ros2 launch custom_navigation custom_nav.launch.py pose_source:=tf offset_x:=-3.807 offset_y:=1.212 goal_standoff:=0.8

world_file defaults to the SAME maze_world.world that Ream's launch file loads
(turtlebot3_gazebo/worlds), so the planner always sees the real walls.
offset_x / offset_y: map = world + offset. Keep 0 if the map frame = Gazebo world frame.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    P = lambda name, typ: ParameterValue(LaunchConfiguration(name), value_type=typ)

    args = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world_file', default_value=PathJoinSubstitution(
            [FindPackageShare('turtlebot3_gazebo'), 'worlds', 'maze_world.world'])),
        DeclareLaunchArgument('offset_x', default_value='0.0'),
        DeclareLaunchArgument('offset_y', default_value='0.0'),
        DeclareLaunchArgument('pose_source', default_value='amcl'),   # 'amcl' or 'tf'
        DeclareLaunchArgument('goal_standoff', default_value='0.0'),  # 0.8 = stop short of a person
    ]

    map_params = {'world_file': P('world_file', str),
                  'map_offset_x': P('offset_x', float),
                  'map_offset_y': P('offset_y', float),
                  'use_sim_time': P('use_sim_time', bool)}

    planner = Node(package='custom_navigation', executable='planner_node', name='ia_star_planner',
                   output='screen',
                   parameters=[{**map_params, 'pose_source': P('pose_source', str),
                                'goal_standoff': P('goal_standoff', float)}])
    smoother = Node(package='custom_navigation', executable='smoother_node', name='path_smoother',
                    output='screen', parameters=[map_params])
    controller = Node(package='custom_navigation', executable='controller_node', name='tb3_controller',
                      output='screen',
                      parameters=[{'pose_source': P('pose_source', str),
                                   'use_sim_time': P('use_sim_time', bool)}])

    return LaunchDescription(args + [controller, smoother, planner])   # controller first: it must not miss the path