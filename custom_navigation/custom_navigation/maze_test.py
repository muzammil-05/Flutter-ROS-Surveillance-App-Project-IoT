"""Test planner + smoother + controller on the REAL world file (no ROS, no Gazebo).
    python3 maze_test.py                      (expects maze_world.world next to this file)
    python3 maze_test.py /path/to/maze_world.world [offset_x offset_y]
Checks: the walls are read correctly, every person spot is reachable from the robot's
possible spawn points, paths are rounded, and the simulated Waffle never touches a wall."""
import math, os, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from world_map import make_map, build_grid
from ia_star import IAStarPlanner, trim_end
from smoother import safe_smoothing
from pure_pursuit import PurePursuitController
from tracking_sim import wall_clearance

world = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "maze_world.world")
off = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.0, 0.0)
RES, INFL, CLEAR, DT, WAFFLE_R = 0.2, 0.50, 0.35, 0.1, 0.22
rects, area = make_map(world, off)
grid, fine = build_grid(rects, area, RES, INFL), build_grid(rects, area, 0.05, INFL)
safe, loose = build_grid(rects, area, 0.05, CLEAR), build_grid(rects, area, 0.05, 0.03)


def free_fn(g):
    def f(x, y):
        i = min(max(int((x - area[0]) / 0.05), 0), g.shape[0] - 1)
        j = min(max(int((y - area[2]) / 0.05), 0), g.shape[1] - 1)
        return not g[i, j]
    return f


def plan(s, g, standoff=0.0):
    _, pruned = IAStarPlanner(s, g, rects, area, RES, 0, INFL, grid, fine).planning()
    if not pruned:
        return None
    pruned = np.array(trim_end(pruned, standoff))
    L = float(np.hypot(*np.diff(pruned, axis=0).T).sum())
    smooth, how = safe_smoothing(pruned, free_fn(safe), max(200, int(L / 0.05) + 1), is_free_end=free_fn(loose))
    return pruned, smooth, how


def drive(path, yaw0, rng):
    c = PurePursuitController(); c.set_path(path)
    x, y = path[0]; yaw = yaw0; traj = [(x, y)]
    for _ in range(12000):
        v, w, done = c.step(x + rng.normal(0, .02), y + rng.normal(0, .02), yaw + rng.normal(0, .02), DT)
        if done and abs(v) < 1e-3 and abs(w) < 1e-3:
            return np.array(traj), True
        x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT; yaw += w * DT
        traj.append((x, y))
    return np.array(traj), False


def run(s, g, rng, standoff=0.0):
    res = plan(s, g, standoff)
    if res is None:
        return None
    pruned, smooth, how = res
    d = smooth[1] - smooth[0]
    traj, ok = drive(smooth, math.atan2(d[1], d[0]), rng)
    return pruned, smooth, how, traj, ok


rng = np.random.default_rng(0)
print(f"world file: {os.path.basename(world)} | {len(rects)} obstacles (walls + boxes) | area x[{area[0]:.1f},{area[1]:.1f}] y[{area[2]:.1f},{area[3]:.1f}] | map offset {off}")

spawns = {"spawn A (model pose)": (3.807, -1.212), "spawn B (saved state)": (-13.024, -1.794)}
people = {"person 1": (-14.0, 0.5), "person 2": (-10.0, 0.5), "person 3": (-6.0, 0.5)}
print("\n--- robot spawn -> person standing still, stopping 0.8 m short of them ---")
shown = None
for sn, sp in spawns.items():
    for pn, pp in people.items():
        # the person's own position (+off) is the goal; planner snaps it if it sits in an inflated zone
        r = run((sp[0] + off[0], sp[1] + off[1]), (pp[0] + off[0], pp[1] + off[1]), rng, standoff=0.8)
        if r is None:
            print(f"{sn} -> {pn}: NO PATH"); continue
        pruned, smooth, how, traj, ok = r
        clr = wall_clearance(traj, rects)
        print(f"{sn} -> {pn}: path {np.hypot(*np.diff(smooth, axis=0).T).sum():5.1f} m | {how:20s} | reached {ok} | closest wall {clr:.2f} m | Waffle hit: {clr < WAFFLE_R}")
        if shown is None and sn.startswith("spawn A") and pn == "person 1":
            shown = (sp, pp, pruned, smooth, traj)

print("\n--- 100 random start/goal pairs inside the maze (the region the robot can reach) ---")
random.seed(3)
from collections import deque
def region_of(start):
    """Flood fill: all free cells connected to `start` (outside the walls is a separate region)."""
    s = (int((start[0] - area[0]) / RES), int((start[1] - area[2]) / RES))
    seen = np.zeros(grid.shape, bool); seen[s] = True; q = deque([s])
    while q:
        a, b = q.popleft()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c, d = a + da, b + db
            if 0 <= c < grid.shape[0] and 0 <= d < grid.shape[1] and not grid[c, d] and not seen[c, d]:
                seen[c, d] = True; q.append((c, d))
    return seen
inside = region_of((3.807 + off[0], -1.212 + off[1]))
def rnd():
    while True:
        q = (random.uniform(area[0] + 1, area[1] - 1), random.uniform(area[2] + 1, area[3] - 1))
        i, j = int((q[0] - area[0]) / RES), int((q[1] - area[2]) / RES)
        if inside[i, j]:
            return q
n = reached = hits = sharp = noplan = 0
for _ in range(100):
    r = run(rnd(), rnd(), rng)
    if r is None:
        noplan += 1; continue
    pruned, smooth, how, traj, ok = r
    n += 1; reached += ok; sharp += how.startswith("fallback")
    hits += wall_clearance(traj, rects) < WAFFLE_R
print(f"planned {n}/100 (no path: {noplan}) | reached goal {reached}/{n} | Waffle wall hits {hits}/{n} | sharp-corner fallbacks {sharp}/{n}")

fig, ax = plt.subplots(figsize=(11, 8))
for (x0, y0, x1, y1) in rects:
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="k"))
ax.imshow(grid.T, origin="lower", extent=[area[0], area[1], area[2], area[3]], alpha=0.12, cmap="Reds")
ax.plot(-17 + off[0], 3.5 + off[1], "c^", ms=12, label="fixed camera (-17, 3.5)")
for pn, pp in people.items():
    ax.plot(pp[0] + off[0], pp[1] + off[1], "mo"); ax.annotate(pn, (pp[0] + off[0], pp[1] + off[1] + .3))
for sn, sp in spawns.items():
    ax.plot(sp[0] + off[0], sp[1] + off[1], "gs", ms=9); ax.annotate(sn, (sp[0] + off[0] + .3, sp[1] + off[1] - .6))
if shown:
    sp, pp, pruned, smooth, traj = shown
    ax.plot(pruned[:, 0], pruned[:, 1], "bo--", ms=4, label="pruned")
    ax.plot(smooth[:, 0], smooth[:, 1], "g-", lw=2, label="smoothed")
    ax.plot(traj[:, 0], traj[:, 1], "m:", lw=1.5, label="simulated robot")
ax.set_aspect("equal"); ax.legend(loc="lower left"); ax.set_title("maze_world.world: real walls read from the file")
plt.tight_layout(); plt.savefig("maze_result.png", dpi=110)
print("\nsaved maze_result.png")