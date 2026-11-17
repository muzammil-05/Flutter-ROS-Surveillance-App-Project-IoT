"""Run:  python3 offline_test.py      (no ROS needed)
Compares  A = the documentation's approach (small margin, plain cubic spline)
     with B = improved approach (bigger margin, rounded corners, clearance-checked),
and simulates the doc's controller to see how close the robot really gets to walls."""
import math, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from world_map import build_rects, build_grid, RAND_AREA
from ia_star import IAStarPlanner
from smoother import arc_length_smoothing, safe_smoothing
from tracking_sim import simulate, wall_clearance

RES = 0.2
BURGER_R, WAFFLE_R = 0.105, 0.22          # TurtleBot3 radii (m); 0.22 = Waffle (Nav2 default footprint)
rects = build_rects()

CONFIGS = {
    "A (doc style)": dict(infl=0.20, fine=False, clear=0.13, rounded=False, points=lambda L: 200),
    "B (improved)":  dict(infl=0.50, fine=True,  clear=0.35, rounded=True,
                          points=lambda L: max(200, int(L / 0.05) + 1)),   # one point every ~5 cm
}
for c in CONFIGS.values():
    c["grid"] = build_grid(rects, RAND_AREA, RES, c["infl"])
    c["fine_grid"] = build_grid(rects, RAND_AREA, 0.05, c["infl"]) if c["fine"] else False
    c["safe"] = build_grid(rects, RAND_AREA, 0.05, c["clear"])


def make_is_free(safe):
    def f(x, y):
        i = min(max(int((x - RAND_AREA[0]) / 0.05), 0), safe.shape[0] - 1)
        j = min(max(int((y - RAND_AREA[2]) / 0.05), 0), safe.shape[1] - 1)
        return not safe[i, j]
    return f


def length(p):
    p = np.asarray(p)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def plan(cfg, s, g):
    pl = IAStarPlanner(s, g, rects, RAND_AREA, RES, 0, cfg["infl"], cfg["grid"], cfg["fine_grid"])
    raw, pruned = pl.planning()
    if not pruned:
        return None
    n = cfg["points"](length(pruned))
    if cfg["rounded"]:
        smooth, how = safe_smoothing(pruned, make_is_free(cfg["safe"]), n)
    else:
        smooth, how = arc_length_smoothing(pruned, n), "plain spline"
    return raw, pruned, smooth


random.seed(1)
pairs = []
g0 = CONFIGS["B (improved)"]["grid"]
# while len(pairs) < 100:
#     pt = lambda: (random.uniform(-19.5, 4.0), random.uniform(-3.5, 3.5))
#     s, g = pt(), pt()
#     ok = lambda q: not g0[int((q[0]-RAND_AREA[0])/RES), int((q[1]-RAND_AREA[2])/RES)]
#     if ok(s) and ok(g):
#         pairs.append((s, g, random.uniform(-0.8, 0.8)))

# --- PURANA RANDOM LOOP HATA KAR YEH LIKHEIN ---
pairs = []

# TEST CASE 1: Start point bilkul deewar ke paas chipka hua hai (Inflation / Nearest Free test)
# Deewar ke paas ka point, coordinates aap map image se dekh kar adjust kar sakte hain.
s1 = (-1.95, 1.95)  
g1 = (2.0, -3.0)
pairs.append((s1, g1, 0.0))  # 0.0 yaw offset hai

# TEST CASE 2: Aisa path jo seedha deewar ko cheerata (cross) hua guzar raha hai
# Taake check ho ke line_free/pruning deewar ko detect karke shortcut rokti hai ya nahi
s2 = (-4.0, 2.0)    # Deewar ke ek taraf
g2 = (4.0, 2.0)     # Deewar ke doosri taraf (beech mein deewar hai)
pairs.append((s2, g2, 0.0))

# TEST CASE 3: Tang rasta (Narrow Corridor / Corner Test)
s3 = (-2.5, -2.5)
g3 = (2.5, 2.5)
pairs.append((s3, g3, 0.4)) # Thoda sa yaw offset de kar simulation check karne ke liye

print(f"Total manual test cases loaded: {len(pairs)}")


best_plot = {}
for name, cfg in CONFIGS.items():
    clr, reached, ms, lens, nb, nw = [], 0, [], [], 0, 0
    for s, g, yaw_off in pairs:
        t0 = time.perf_counter()
        res = plan(cfg, s, g)
        ms.append((time.perf_counter() - t0) * 1000)
        if res is None:
            continue
        raw, pruned, smooth = res
        lens.append(length(smooth))
        d = smooth[1] - smooth[0]
        traj, ok = simulate(smooth, math.atan2(d[1], d[0]) + yaw_off)
        reached += ok
        c = wall_clearance(traj, rects)
        clr.append(c)
        nb += c < BURGER_R
        nw += c < WAFFLE_R
        # if name not in best_plot and length(pruned) > 25:
        #     best_plot[name] = (s, g, raw, pruned, smooth, traj)
        if name not in best_plot:
            best_plot[name] = (s, g, raw, pruned, smooth, traj)

    clr = np.array(clr)
    print(f"\n=== {name} ===")
    print(f"avg smooth length {np.mean(lens):.2f} m | plan time mean {np.mean(ms):.0f} ms")
    print(f"robot reached goal: {reached}/{len(pairs)}")
    print(f"closest the ROBOT CENTRE gets to a wall surface: min {clr.min():.3f} m, 5th percentile {np.percentile(clr,5):.3f} m, median {np.median(clr):.3f} m")
    print(f"would hit a wall -> burger (r=0.105): {nb}/{len(pairs)} runs | waffle (r=0.22): {nw}/{len(pairs)} runs")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, (name, (s, g, raw, pruned, smooth, traj)) in zip(axes, best_plot.items()):
    for (x0, y0, x1, y1) in rects:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="k"))
    ax.plot(*zip(*pruned), "bo--", ms=4, label="pruned waypoints")
    ax.plot(smooth[:, 0], smooth[:, 1], "g-", lw=2, label="smoothed path")
    ax.plot(traj[:, 0], traj[:, 1], "m:", lw=1.5, label="simulated robot")
    ax.plot(*s, "ks"); ax.plot(*g, "k*", ms=12)
    ax.set_xlim(-6, 5); ax.set_ylim(-4.2, 4.2); ax.set_aspect("equal"); ax.legend(loc="upper left")
    ax.set_title(name)
plt.tight_layout(); plt.savefig("result.png", dpi=110)
print("\nsaved result.png (zoomed on the room)")