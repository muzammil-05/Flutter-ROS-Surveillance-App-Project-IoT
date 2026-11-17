"""Pure-pursuit path tracker for a differential-drive robot (TurtleBot3 Waffle).

ROS-free on purpose: controller_node.py only does the ROS plumbing, so this maths
can be tested offline (see controller_test.py).

Frames: all poses/paths are in the `map` frame, metres and radians.
Idea (pure pursuit): pick a point `L` metres ahead ON the path, then drive along the
circle that goes through that point:   curvature = 2*y_r / L^2,   w = v * curvature
(x_r, y_r = the point expressed in the robot frame). Extra rules for smooth motion:
  * speed and turn rate change gradually (acceleration limits, no jerks)
  * slow down in tight curves and before the goal (so the robot stops softly)
  * rotate in place first if the path is far off to the side
  * progress along the path only moves forward (no stalling, no jumping legs)
"""
import math
import numpy as np


def normalize_angle(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def slew(prev, target, max_step):
    """Move `prev` towards `target` by at most `max_step` (acceleration limiter)."""
    return prev + max(-max_step, min(max_step, target - prev))


class PurePursuitController:
    def __init__(self,
                 v_max=0.20,            # cruise speed [m/s]   (Waffle limit is 0.26)
                 v_min=0.04,            # lowest speed while still driving [m/s]
                 w_max=1.2,             # max turn rate [rad/s] (Waffle limit is 1.82)
                 a_max=0.15,            # max linear accel/decel [m/s^2]
                 alpha_max=1.5,         # max angular accel [rad/s^2]
                 a_lat_max=0.10,        # max sideways accel in curves [m/s^2]
                 lookahead_min=0.30,    # lookahead distance at standstill [m]
                 lookahead_gain=1.0,    # extra lookahead per m/s of speed [s]
                 goal_tolerance=0.12,   # stop within this distance of the goal [m]
                 rotate_enter=math.radians(50),   # start rotating in place above this
                 rotate_exit=math.radians(20),    # stop rotating in place below this
                 k_rotate=1.5,          # turn-in-place gain
                 search_window=3.0):    # how far ahead (m) we look for the nearest path point
        self.v_max, self.v_min, self.w_max = v_max, v_min, w_max
        self.a_max, self.alpha_max, self.a_lat_max = a_max, alpha_max, a_lat_max
        self.l_min, self.l_gain = lookahead_min, lookahead_gain
        self.goal_tol = goal_tolerance
        self.rot_in, self.rot_out, self.k_rot = rotate_enter, rotate_exit, k_rotate
        self.window = search_window
        self.path = None
        self.s = None
        self.idx = 0
        self.v = 0.0            # last commanded speed
        self.w = 0.0            # last commanded turn rate
        self.rotating = False
        self.done = True
        self._fresh = False     # True right after set_path()

    # ------------------------------------------------------------------ path
    def set_path(self, points):
        """Start tracking a new path (N x 2 array in map frame). The current speed is
        kept, so a path change in mid-motion does not cause a jerk."""
        pts = np.asarray(points, dtype=float)
        if len(pts) < 2:
            return False
        ds = np.hypot(*np.diff(pts, axis=0).T)
        self.path = pts
        self.s = np.insert(np.cumsum(ds), 0, 0.0)    # arc length of every point
        self.idx = 0
        self.rotating = False
        self.done = False
        self._fresh = True
        return True

    def _nearest_forward(self, x, y, first_call):
        """Closest path point, searched only FORWARD from the last one (progress never
        goes backwards, so a U-shaped path cannot trick us onto the wrong leg)."""
        n = len(self.path)
        if first_call:
            lo, hi = 0, n                             # new path: search everywhere
        else:
            lo = self.idx
            hi = int(np.searchsorted(self.s, self.s[lo] + self.window, side="right"))
            hi = min(max(hi, lo + 1), n)
        seg = self.path[lo:hi]
        self.idx = lo + int(np.argmin(np.hypot(seg[:, 0] - x, seg[:, 1] - y)))

    # ------------------------------------------------------------------ step
    def step(self, x, y, yaw, dt):
        """One control cycle. Returns (v, w, done)."""
        if self.path is None:
            return self._command(0.0, 0.0, dt, done=True)

        self._nearest_forward(x, y, first_call=self._fresh)
        self._fresh = False
        goal = self.path[-1]
        dist_goal = math.hypot(goal[0] - x, goal[1] - y)

        if self.done or dist_goal < self.goal_tol:
            self.done = True
            return self._command(0.0, 0.0, dt, done=True)

        # --- lookahead point: L metres further along the path ---
        L = self.l_min + self.l_gain * abs(self.v)
        j = min(int(np.searchsorted(self.s, self.s[self.idx] + L)), len(self.path) - 1)
        tx, ty = self.path[j]
        dx, dy = tx - x, ty - y
        xr = math.cos(yaw) * dx + math.sin(yaw) * dy       # target in robot frame
        yr = -math.sin(yaw) * dx + math.cos(yaw) * dy
        Ld = max(math.hypot(xr, yr), 1e-6)
        alpha = math.atan2(yr, xr)                          # heading error to target

        # --- rotate in place when the target is far off to the side (with hysteresis) ---
        if abs(alpha) > self.rot_in:
            self.rotating = True
        elif abs(alpha) < self.rot_out:
            self.rotating = False
        if self.rotating:
            w_t = max(-self.w_max, min(self.w_max, self.k_rot * alpha))
            return self._command(0.0, w_t, dt, done=False)

        # --- pure pursuit: circle through the lookahead point ---
        kappa = 2.0 * yr / (Ld * Ld)
        v_t = self.v_max
        if abs(kappa) > 1e-6:
            v_t = min(v_t, self.w_max / abs(kappa))                  # turn rate stays <= w_max
            v_t = min(v_t, math.sqrt(self.a_lat_max / abs(kappa)))   # comfortable sideways accel
        v_t *= max(0.4, math.cos(alpha))                             # slower if still misaligned
        remaining = (self.s[-1] - self.s[self.idx]) + math.hypot(self.path[self.idx][0] - x,
                                                                  self.path[self.idx][1] - y)
        v_stop = math.sqrt(2.0 * self.a_max * max(remaining - self.goal_tol, 0.0))
        v_t = min(v_t, max(self.v_min, v_stop))                      # brake softly before the goal
        v_t = max(v_t, self.v_min)

        v_cmd = slew(self.v, v_t, self.a_max * dt)                   # acceleration limit
        w_t = max(-self.w_max, min(self.w_max, v_cmd * kappa))
        return self._command(v_cmd, w_t, dt, done=False, v_already_limited=True)

    def _command(self, v_t, w_t, dt, done, v_already_limited=False):
        v = v_t if v_already_limited else slew(self.v, v_t, self.a_max * dt)
        w = slew(self.w, w_t, self.alpha_max * dt)
        self.v, self.w = v, w
        return v, w, done