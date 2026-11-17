"""IA* (Informed A*) planner on a grid. Same call style as the project doc:
    raw, pruned = IAStarPlanner(start, goal, obstacleList_rect, randArea,
                                resolution=0.2, end_point=0).planning()
Returns two lists of (x, y) in metres. Empty lists mean "no path"."""
import heapq
import math
import numpy as np
try:
    from .world_map import build_grid      # when run as a ROS package
except ImportError:
    from world_map import build_grid       # when run standalone (offline_test.py)

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


class IAStarPlanner:
    def __init__(self, start, goal, obstacleList_rect, randArea,
                 resolution=0.2, end_point=0, inflation=0.20, grid=None,
                 fine_grid=None, fine_res=0.05):
        self.start, self.goal = tuple(start[:2]), tuple(goal[:2])
        self.rand_area = randArea
        self.res = resolution
        self.end_point = end_point          # kept for compatibility with the doc
        # Pass a pre-built grid to avoid rebuilding it on every plan.
        self.grid = grid if grid is not None else build_grid(
            obstacleList_rect, randArea, resolution, inflation)
        self.nx, self.ny = self.grid.shape
        # Fine grid (5 cm) used ONLY for line-of-sight pruning, so shortcuts never
        # cut closer to a wall than `inflation`. fine_grid=False -> use coarse grid.
        self.fine_res = fine_res
        if fine_grid is None:
            fine_grid = build_grid(obstacleList_rect, randArea, fine_res, inflation)
        self.fine = None if fine_grid is False else fine_grid

    # ---- coordinate helpers -------------------------------------------------
    def to_cell(self, x, y):
        i = int((x - self.rand_area[0]) / self.res)
        j = int((y - self.rand_area[2]) / self.res)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def to_world(self, i, j):
        return (self.rand_area[0] + (i + 0.5) * self.res,
                self.rand_area[2] + (j + 0.5) * self.res)

    def in_bounds(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def nearest_free(self, cell, max_r=10):
        """If start/goal fall inside an inflated zone, snap to nearest free cell."""
        i, j = cell
        if not self.grid[i, j]:
            return cell
        for r in range(1, max_r + 1):
            best = None
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    a, b = i + di, j + dj
                    if self.in_bounds(a, b) and not self.grid[a, b]:
                        d = math.hypot(di, dj)
                        if best is None or d < best[0]:
                            best = (d, (a, b))
            if best:
                return best[1]
        return None

    # ---- search -------------------------------------------------------------
    @staticmethod
    def heuristic(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)     # octile distance

    def search(self, s, g):
        open_heap = [(self.heuristic(s, g), 0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        while open_heap:
            _, c, cur = heapq.heappop(open_heap)
            if cur == g:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1]
            if c > cost[cur]:
                continue
            for di, dj, step in NEIGHBOURS:
                n = (cur[0] + di, cur[1] + dj)
                if not self.in_bounds(*n) or self.grid[n]:
                    continue
                # no corner cutting on diagonal moves
                if di and dj and (self.grid[cur[0] + di, cur[1]] or
                                  self.grid[cur[0], cur[1] + dj]):
                    continue
                nc = c + step
                if nc < cost.get(n, 1e18):
                    cost[n], came[n] = nc, cur
                    h = self.heuristic(n, g)
                    # tiny tie-break towards the goal -> fewer expanded nodes
                    heapq.heappush(open_heap, (nc + h * 1.0001, nc, n))
        return None

    def line_free(self, p, q):
        """Sampled line-of-sight check on the inflated grid (world coordinates)."""
        g, res = (self.fine, self.fine_res) if self.fine is not None else (self.grid, self.res)
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(d / (res * 0.5)) + 1)
        for k in range(n + 1):
            t = k / n
            x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
            i = min(max(int((x - self.rand_area[0]) / res), 0), g.shape[0] - 1)
            j = min(max(int((y - self.rand_area[2]) / res), 0), g.shape[1] - 1)
            if g[i, j]:
                return False
        return True

    def prune(self, pts):
        """Greedy line-of-sight pruning: drop waypoints that are not needed."""
        if len(pts) < 3:
            return pts
        out, anchor = [pts[0]], 0
        while anchor < len(pts) - 1:
            far = len(pts) - 1
            while far > anchor + 1 and not self.line_free(pts[anchor], pts[far]):
                far -= 1
            out.append(pts[far])
            anchor = far
        return out

    def planning(self):
        s = self.nearest_free(self.to_cell(*self.start))
        g = self.nearest_free(self.to_cell(*self.goal))
        if s is None or g is None:
            return [], []
        cells = self.search(s, g)
        if cells is None:
            return [], []
        raw = [self.to_world(i, j) for i, j in cells]
        raw[0], raw[-1] = self.start, self.goal      # exact start / goal
        return raw, self.prune(raw)


def trim_end(path, standoff):
    """Cut `standoff` metres off the END of a path (list of (x, y)). Used so the robot stops
    in front of a person instead of driving into them. Paths shorter than 1.5x the
    standoff are left alone (the robot is already close enough)."""
    pts = [tuple(p) for p in path]
    if standoff <= 0 or len(pts) < 2:
        return pts
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:]))
    if total < 1.5 * standoff:
        return pts
    left = standoff
    while len(pts) > 1:
        a, b = pts[-2], pts[-1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg > left:
            t = (seg - left) / seg
            pts[-1] = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            return pts
        left -= seg
        pts.pop()
    return pts