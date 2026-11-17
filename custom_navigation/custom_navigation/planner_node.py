"""Stage 1: IA* planner node.   start + /goal_pose  ->  /planned_path
Start pose comes from /amcl_pose (pose_source:=amcl) or TF map->base_footprint (pose_source:=tf).
The walls are read from the Gazebo world file (parameter world_file), so the planner always
matches the world Ream launches.  (Only checked with a ROS-free simulation, see the README notes.)"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException

try:
    from .world_map import make_map, build_grid
    from .ia_star import IAStarPlanner, trim_end
except ImportError:
    from world_map import make_map, build_grid
    from ia_star import IAStarPlanner, trim_end


class IAStarPlannerNode(Node):
    def __init__(self):
        super().__init__('ia_star_planner')
        self.declare_parameter('world_file', '')            # path to maze_world.world ('' = old sim_world walls)
        self.declare_parameter('map_offset_x', 0.0)         # map = world + offset (ask Ream)
        self.declare_parameter('map_offset_y', 0.0)
        self.declare_parameter('resolution', 0.2)
        self.declare_parameter('inflation', 0.50)           # m kept from walls (Waffle radius 0.22 + margin)
        self.declare_parameter('pose_source', 'amcl')       # 'amcl' or 'tf'
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('goal_standoff', 0.0)        # m to stop short of the goal (0.8 for person goals)
        g = lambda n: self.get_parameter(n).value

        self.res, self.infl = g('resolution'), g('inflation')
        self.standoff = g('goal_standoff')
        self.pose_source, self.base_frame = g('pose_source'), g('base_frame')
        self.rects, self.rand_area = make_map(
            g('world_file'), (g('map_offset_x'), g('map_offset_y')))
        self.grid = build_grid(self.rects, self.rand_area, self.res, self.infl)       # built once
        self.fine_grid = build_grid(self.rects, self.rand_area, 0.05, self.infl)      # for safe shortcuts
        self.start = None
        self.goal = None

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        if self.pose_source == 'amcl':
            self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_cb, 10)
        else:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        a = self.rand_area
        self.get_logger().info(
            f"IA* planner ready: {len(self.rects)} obstacles, area x[{a[0]:.1f},{a[1]:.1f}] "
            f"y[{a[2]:.1f},{a[3]:.1f}], pose source: {self.pose_source}")

    def pose_cb(self, msg):
        p = msg.pose.pose.position
        self.start = (p.x, p.y)

    def update_start_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform('map', self.base_frame, Time())
        except TransformException:
            return
        self.start = (t.transform.translation.x, t.transform.translation.y)

    def goal_cb(self, msg):
        if msg.header.frame_id not in ('map', ''):
            self.get_logger().warn(f"Goal frame is '{msg.header.frame_id}', expected 'map'")
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.plan_path()

    def plan_path(self):
        if self.pose_source == 'tf':
            self.update_start_from_tf()
        if self.start is None or self.goal is None:
            self.get_logger().warn('Need both a start pose and a goal')
            return
        a = self.rand_area
        if not (a[0] <= self.goal[0] <= a[1] and a[2] <= self.goal[1] <= a[3]):
            self.get_logger().error(f'Goal {self.goal} is outside the map area, ignoring it '
                                    '(wrong frame/offset?)')
            return
        self.get_logger().info(f'Planning {self.start} -> {self.goal}')
        planner = IAStarPlanner(self.start, self.goal, self.rects, self.rand_area,
                                self.res, 0, self.infl, self.grid, self.fine_grid)
        raw, pruned = planner.planning()
        if not pruned:
            self.get_logger().error('No path found (goal inside a wall or unreachable)')
            return
        pruned = trim_end(pruned, self.standoff)
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in pruned:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)
        self.get_logger().info(f'Published {len(path.poses)} waypoints')


def main(args=None):
    rclpy.init(args=args)
    node = IAStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()