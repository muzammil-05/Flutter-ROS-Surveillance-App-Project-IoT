"""Arc-length cubic-spline smoothing (doc Listing 10.5) with safety fixes."""
import numpy as np
from scipy.interpolate import CubicSpline


def arc_length_smoothing(pts, num_points=200):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    # drop repeated points: CubicSpline needs strictly increasing s
    keep = np.insert(np.hypot(*np.diff(pts, axis=0).T) > 1e-9, 0, True)
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    ds = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.insert(np.cumsum(ds), 0, 0.0)
    s_new = np.linspace(0, s[-1], num_points)
    if len(pts) == 2:                       # straight line
        t = (s_new / s[-1])[:, None]
        return pts[0] + t * (pts[1] - pts[0])
    bc = "not-a-knot" if len(pts) > 3 else "natural"
    cs_x = CubicSpline(s, pts[:, 0], bc_type=bc)
    cs_y = CubicSpline(s, pts[:, 1], bc_type=bc)
    return np.column_stack((cs_x(s_new), cs_y(s_new)))


def densify(pts, max_seg):
    """Insert points so no segment is longer than max_seg."""
    pts = np.asarray(pts, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / max_seg)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def round_corners(pts, radius, samples=12):
    """Keep straight segments straight; replace each corner by a short smooth curve
    (quadratic Bezier) that starts/ends `radius` metres from the corner. The curve
    always stays INSIDE the turn, so it can never swing wide the way a spline does."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3 or radius <= 0:
        return pts
    out = [pts[0]]
    for k in range(1, len(pts) - 1):
        a, p, b = pts[k - 1], pts[k], pts[k + 1]
        la, lb = np.hypot(*(a - p)), np.hypot(*(b - p))
        if la < 1e-9 or lb < 1e-9:
            continue
        d = min(radius, 0.5 * la, 0.5 * lb)
        q0 = p + (a - p) / la * d
        q1 = p + (b - p) / lb * d
        for t in np.linspace(0, 1, samples):
            out.append((1 - t) ** 2 * q0 + 2 * (1 - t) * t * p + t ** 2 * q1)
    out.append(pts[-1])
    return np.array(out)


def safe_smoothing(pruned, is_free, num_points=200, radii=(1.0, 0.7, 0.5, 0.35)):
    """Round the corners as much as possible while the curve stays clear of walls
    (is_free(x, y) must include the robot radius + safety margin), then resample with
    the arc-length cubic spline from the documentation. Returns (path, description)."""
    pruned = np.asarray(pruned, dtype=float)
    for r in radii:
        smooth = arc_length_smoothing(densify(round_corners(pruned, r), 0.05), num_points)
        if all(is_free(x, y) for x, y in smooth):
            return smooth, f"corner radius {r} m"
    return densify(pruned, 0.1), "fallback: straight segments (no rounding is safe)"