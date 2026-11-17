"""Offline test of pure_pursuit.py (no ROS). Compares it with the controller from the
project document on the same 100 planned + smoothed paths.
Run:  python3 controller_test.py"""
import math, random
import numpy as np
from world_map import build_rects, build_grid, RAND_AREA
from ia_star import IAStarPlanner
from smoother import safe_smoothing
from pure_pursuit import PurePursuitController
from tracking_sim import simulate as doc_simulate, wall_clearance

RES, INFL, CLEAR, DT, WAFFLE_R = 0.2, 0.50, 0.35, 0.1, 0.22
rects = build_rects()
grid = build_grid(rects, RAND_AREA, RES, INFL)
fine = build_grid(rects, RAND_AREA, 0.05, INFL)
safe = build_grid(rects, RAND_AREA, 0.05, CLEAR)


def is_free(x, y):
    i = min(max(int((x - RAND_AREA[0]) / 0.05), 0), safe.shape[0] - 1)
    j = min(max(int((y - RAND_AREA[2]) / 0.05), 0), safe.shape[1] - 1)
    return not safe[i, j]


def plan_and_smooth(s, g):
    _, pruned = IAStarPlanner(s, g, rects, RAND_AREA, RES, 0, INFL, grid, fine).planning()
    if not pruned:
        return None
    L = float(np.hypot(*np.diff(np.array(pruned), axis=0).T).sum())
    return safe_smoothing(pruned, is_free, max(200, int(L / 0.05) + 1))[0]


def run_pp(path, yaw0, rng, params):
    """Closed loop: unicycle robot + noisy pose (AMCL-like) + our controller."""
    c = PurePursuitController(**params)
    c.set_path(path)
    x, y = path[0]; yaw = yaw0
    traj, vs = [(x, y)], [0.0]
    for _ in range(6000):
        px, py, pyaw = x + rng.normal(0, .02), y + rng.normal(0, .02), yaw + rng.normal(0, .02)
        v, w, done = c.step(px, py, pyaw, DT)
        if done and abs(v) < 1e-3 and abs(w) < 1e-3:
            return np.array(traj), np.array(vs), True
        x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT; yaw += w * DT
        traj.append((x, y)); vs.append(v)
    return np.array(traj), np.array(vs), False


def accel_stats(traj):
    v = np.hypot(*np.diff(traj, axis=0).T) / DT
    v = np.concatenate(([0.0], v, [0.0]))      # robot starts and ends at rest
    a = np.abs(np.diff(v)) / DT
    return a.max(), float(np.percentile(a, 95))


def cross_track(traj, path):
    return max(float(np.min(np.hypot(path[:, 0] - px, path[:, 1] - py))) for px, py in traj[::5])


if __name__ == "__main__":
    random.seed(1); rng = np.random.default_rng(0)
    pairs = []
    while len(pairs) < 100:
        pt = lambda: (random.uniform(-19.5, 4.0), random.uniform(-3.5, 3.5))
        s, g = pt(), pt()
        ok = lambda q: not grid[int((q[0] - RAND_AREA[0]) / RES), int((q[1] - RAND_AREA[2]) / RES)]
        if ok(s) and ok(g):
            pairs.append((s, g, random.uniform(-0.8, 0.8)))
    rows = {"doc": [], "pp": []}
    for s, g, yo in pairs:
        path = plan_and_smooth(s, g)
        if path is None:
            continue
        d = path[1] - path[0]
        yaw0 = math.atan2(d[1], d[0]) + yo
        tr, ok = doc_simulate(path, yaw0)
        am, a95 = accel_stats(tr)
        rows["doc"].append((ok, wall_clearance(tr, rects), len(tr) * DT, am, a95, cross_track(tr, path)))
        tr, vs, ok = run_pp(path, yaw0, rng, {})
        am, a95 = accel_stats(tr)
        rows["pp"].append((ok, wall_clearance(tr, rects), len(tr) * DT, am, a95, cross_track(tr, path)))
    for k, name in (("doc", "document controller"), ("pp", "pure_pursuit.py (new)")):
        a = np.array(rows[k], dtype=float)
        print(f"\n=== {name} ===")
        print(f"reached goal         : {int(a[:,0].sum())}/{len(a)}")
        print(f"closest wall approach: min {a[:,1].min():.3f} m | Waffle hits (r={WAFFLE_R}): {int((a[:,1] < WAFFLE_R).sum())}/{len(a)}")
        print(f"mean time to goal    : {a[:,2].mean():.0f} s")
        print(f"acceleration         : worst {a[:,3].max():.2f} m/s^2, 95th pct {a[:,4].mean():.3f} m/s^2")
        print(f"max distance from path: {a[:,5].max():.2f} m")