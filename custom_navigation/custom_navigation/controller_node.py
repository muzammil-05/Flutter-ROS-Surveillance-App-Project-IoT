#!/usr/bin/env python3
"""Stage 3: controller node.  /smoothed_path + robot pose  ->  /cmd_vel

Subscribes : /smoothed_path (nav_msgs/Path, frame 'map')
             /amcl_pose     (geometry_msgs/PoseWithCovarianceStamped)   [pose_source:=amcl]
             or TF map -> base_footprint                                  [pose_source:=tf]
Publishes  : /cmd_vel (geometry_msgs/Twist)   (Humble TurtleBot3 uses plain Twist)

All the maths lives in pure_pursuit.py (ROS-free, tested offline). This file is only
the ROS plumbing: read the pose, call the controller at a fixed rate, publish the command.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener, TransformException

try:                                    # works as a ROS package ...
    from .pure_pursuit import PurePursuitController
except ImportError:                     # ... or when run directly from the folder
    from pure_pursuit import PurePursuitController


def yaw_from_quaternion(q):
    """Yaw (rotation about z) from a quaternion; enough for a planar robot."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ControllerNode(Node):
    def __init__(self):
        super().__init__('tb3_controller')

        # ---------------- parameters (change from launch file or command line) ----------------
        self.declare_parameter('pose_source', 'amcl')      # 'amcl' or 'tf'
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('control_rate', 10.0)       # Hz
        self.declare_parameter('v_max', 0.20)              # m/s   (Waffle max 0.26)
        self.declare_parameter('w_max', 1.2)               # rad/s (Waffle max 1.82)
        self.declare_parameter('a_max', 0.15)              # m/s^2
        self.declare_parameter('alpha_max', 1.5)           # rad/s^2
        self.declare_parameter('lookahead_min', 0.30)      # m
        self.declare_parameter('lookahead_gain', 1.0)      # s
        self.declare_parameter('goal_tolerance', 0.12)     # m
        self.declare_parameter('pose_timeout', 2.0)        # s, stop if the pose is older (tf mode)

        g = lambda name: self.get_parameter(name).value
        self.pose_source = g('pose_source')
        self.base_frame = g('base_frame')
        self.pose_timeout = g('pose_timeout')
        self.ctrl = PurePursuitController(
            v_max=g('v_max'), w_max=g('w_max'), a_max=g('a_max'), alpha_max=g('alpha_max'),
            lookahead_min=g('lookahead_min'), lookahead_gain=g('lookahead_gain'),
            goal_tolerance=g('goal_tolerance'))

        # ---------------- state ----------------
        self.pose = None            # (x, y, yaw) in map frame
        self.pose_time = None       # ROS time of the last pose
        self.active = False         # True while we are driving a path
        self.last_time = self.get_clock().now()

        # ---------------- ROS interfaces ----------------
        self.create_subscription(Path, '/smoothed_path', self.path_cb, 10)
        if self.pose_source == 'amcl':
            self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)
        else:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / g('control_rate'), self.control_loop)
        self.get_logger().info(f"Controller ready (pose source: {self.pose_source}); waiting for /smoothed_path")

    # ---------------- callbacks ----------------
    def amcl_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        self.pose_time = self.get_clock().now()

    def path_cb(self, msg):
        if len(msg.poses) < 2:
            self.get_logger().warn('Received a path with fewer than 2 points, ignoring it')
            return
        if msg.header.frame_id not in ('map', ''):
            self.get_logger().warn(f"Path frame is '{msg.header.frame_id}', expected 'map'")
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if self.ctrl.set_path(pts):
            self.active = True
            self.last_time = self.get_clock().now()
            self.get_logger().info(f'New path with {len(pts)} points, driving to ({pts[-1][0]:.2f}, {pts[-1][1]:.2f})')

    # ---------------- pose ----------------
    def update_pose_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform('map', self.base_frame, Time())
        except TransformException:
            return
        tr, q = t.transform.translation, t.transform.rotation
        self.pose = (tr.x, tr.y, yaw_from_quaternion(q))
        self.pose_time = self.get_clock().now()

    # ---------------- main loop ----------------
    def control_loop(self):
        if not self.active:
            return                                   # idle: do not fight other /cmd_vel sources
        if self.pose_source == 'tf':
            self.update_pose_from_tf()
        if self.pose is None:
            self.get_logger().warn('No robot pose yet', throttle_duration_sec=2.0)
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        dt = min(max(dt, 0.01), 0.2)                 # protect the acceleration limiter

        # AMCL only publishes after the robot moved, so an old pose is normal there.
        # With TF the pose is continuous, so an old pose really means something broke.
        stale = (self.pose_source == 'tf' and
                 (now - self.pose_time).nanoseconds * 1e-9 > self.pose_timeout)
        if stale:
            self.get_logger().warn('Pose is too old, stopping', throttle_duration_sec=2.0)
            self.publish(0.0, 0.0)
            return

        x, y, yaw = self.pose
        v, w, done = self.ctrl.step(x, y, yaw, dt)
        self.publish(v, w)
        if done and abs(v) < 1e-3 and abs(w) < 1e-3:
            self.active = False                      # goal reached and stopped
            self.get_logger().info('Goal reached, robot stopped')

    def publish(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.publish(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()                              # never leave the robot moving
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()