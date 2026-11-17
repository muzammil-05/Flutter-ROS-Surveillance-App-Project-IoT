"""Kinematic simulation of the controller in the project doc (Listing 10.6/10.7).
Answers: 'if the robot follows this path, how close does it really get to a wall?'
Not Gazebo: no wheel slip, no AMCL noise, so treat results as optimistic."""
import math
import numpy as np


def simulate(path, start_yaw, dt=0.1, max_steps=4000):
    path = np.asarray(path, dtype=float)
    v_max, v_min, w_max = 0.20, 0.05, 1.2
    k_heading, k_damping = 1.8, 0.4
    la_min, la_max, goal_tol = 4, 10, 0.12
    x, y = path[0]; yaw = start_yaw
    goal = path[-1]; idx = 0; prev_w = 0.0
    traj = [(x, y)]
    for _ in range(max_steps):
        dist_goal = np.hypot(goal[0] - x, goal[1] - y)
        if dist_goal < goal_tol:
            return np.array(traj), True
        while idx < len(path) - 1 and np.hypot(path[idx][0] - x, path[idx][1] - y) < 0.15:
            idx += 1
        la = int(la_min + (la_max - la_min) * min(1.0, dist_goal / 2.0))
        tx, ty = path[min(idx + la, len(path) - 1)]
        err = math.atan2(ty - y, tx - x) - yaw
        err = (err + math.pi) % (2 * math.pi) - math.pi
        w = max(-w_max, min(w_max, k_heading * err - k_damping * prev_w))
        v = max(v_min, min(v_max, v_max * (1.0 - 0.6 * abs(err) / math.pi)))
        prev_w = w
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        traj.append((x, y))
    return np.array(traj), False


def wall_clearance(traj, rects):
    """Smallest distance from robot centre to any wall surface along the trajectory."""
    X, Y = traj[:, 0][:, None], traj[:, 1][:, None]
    r = np.array(rects)
    dx = np.maximum(np.maximum(r[:, 0] - X, 0), X - r[:, 2])
    dy = np.maximum(np.maximum(r[:, 1] - Y, 0), Y - r[:, 3])
    return float(np.hypot(dx, dy).min())