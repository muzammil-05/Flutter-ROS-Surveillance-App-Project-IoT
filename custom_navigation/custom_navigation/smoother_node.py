"""Stage 2: /planned_path -> /smoothed_path (corner rounding + arc-length cubic spline).
Uses the SAME world_file and map offset as planner_node, so both see the same walls."""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

try:
    from .world_map import make_map, build_grid
    from .smoother import safe_smoothing
except ImportError:
    from world_map import make_map, build_grid
    from smoother import safe_smoothing


class PathSmoother(Node):
    def __init__(self):
        super().__init__('path_smoother')
        self.declare_parameter('world_file', '')             # same value as for planner_node
        self.declare_parameter('map_offset_x', 0.0)
        self.declare_parameter('map_offset_y', 0.0)
        self.declare_parameter('robot_clearance', 0.35)      # must exceed robot radius (Waffle ~0.22)
        self.declare_parameter('point_spacing', 0.05)        # metres between path points
        g = lambda n: self.get_parameter(n).value
        rects, self.area = make_map(g('world_file'), (g('map_offset_x'), g('map_offset_y')))
        self.cell = 0.05
        self.safe_grid = build_grid(rects, self.area, self.cell, g('robot_clearance'))
        # loose grid ("just not inside a wall") for the first/last 0.8 m of the path
        self.end_grid = build_grid(rects, self.area, self.cell, 0.03)
        self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        self.pub = self.create_publisher(Path, '/smoothed_path', 10)
        self.get_logger().info(f'Smoother ready with {len(rects)} obstacles')

    def _free(self, grid, x, y):
        i = min(max(int((x - self.area[0]) / self.cell), 0), grid.shape[0] - 1)
        j = min(max(int((y - self.area[2]) / self.cell), 0), grid.shape[1] - 1)
        return not grid[i, j]

    def is_free(self, x, y):
        return self._free(self.safe_grid, x, y)

    def is_free_end(self, x, y):
        return self._free(self.end_grid, x, y)

    def path_cb(self, msg):
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        if len(pts) < 2:
            return
        length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        n = max(200, int(length / self.get_parameter('point_spacing').value) + 1)
        smooth, how = safe_smoothing(pts, self.is_free, n, is_free_end=self.is_free_end)
        out = Path()
        out.header = msg.header
        for x, y in smooth:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            out.poses.append(ps)
        self.pub.publish(out)
        self.get_logger().info(f'Smoothed path published ({how})')


def main(args=None):
    rclpy.init(args=args)
    node = PathSmoother()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()