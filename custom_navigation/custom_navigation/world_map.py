"""Manual obstacle map of sim_world (from the project documentation, Listing 10.1).
Pure Python + numpy: no ROS needed, so you can develop and test it anywhere."""
import numpy as np

WALL_THICKNESS = 0.15
RAND_AREA = [-20.0, 4.5, -4.0, 4.0]   # [min_x, max_x, min_y, max_y]

WALLS = [
    # Outer boundary
    ((4.5, 4), (4.5, -4)),      # right
    ((4.5, 4), (-20, 4)),       # top
    ((-20, 4), (-20, -4)),      # left
    ((-20, -4), (4.5, -4)),     # bottom
    # Inner walls (room structure)
    ((-5, 4), (-5, -2.5)),
    ((-5, -2.5), (2.2, -2.5)),
    ((2.2, -2.5), (2.2, 1.9)),
    ((2.2, 1.9), (-2, 1.9)),
    ((-2, 1.9), (-2, -1)),
]


def build_rects(walls=WALLS, thickness=WALL_THICKNESS):
    """Each wall segment -> axis-aligned rectangle (xmin, ymin, xmax, ymax)."""
    rects = []
    h = thickness / 2.0
    for (x1, y1), (x2, y2) in walls:
        rects.append((min(x1, x2) - h, min(y1, y2) - h,
                      max(x1, x2) + h, max(y1, y2) + h))
    return rects


def build_grid(rects, rand_area=RAND_AREA, resolution=0.2, inflation=0.20):
    """Occupancy grid, True = blocked. A cell is blocked if its centre is within
    `inflation` metres of any wall (inflation ~ robot radius + safety margin;
    TurtleBot3 burger radius is about 0.105 m)."""
    min_x, max_x, min_y, max_y = rand_area
    nx = int(np.ceil((max_x - min_x) / resolution))
    ny = int(np.ceil((max_y - min_y) / resolution))
    xs = min_x + (np.arange(nx) + 0.5) * resolution
    ys = min_y + (np.arange(ny) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys, indexing="ij")          # shape (nx, ny)
    blocked = np.zeros((nx, ny), dtype=bool)
    for (x0, y0, x1, y1) in rects:
        dx = np.maximum(np.maximum(x0 - X, 0.0), X - x1)
        dy = np.maximum(np.maximum(y0 - Y, 0.0), Y - y1)
        blocked |= (np.hypot(dx, dy) <= inflation)
    return blocked


def point_in_rects(x, y, rects):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for (x0, y0, x1, y1) in rects)


# ----------------------------------------------------------------------------------------
# Read the walls straight from the Gazebo world file (so the planner always matches the
# world Ream launches, instead of hard-coding coordinates by hand).
# ----------------------------------------------------------------------------------------
import math
import xml.etree.ElementTree as ET

_SKIP_MODELS = ("ground_plane", "turtlebot3", "fixed_camera", "sun")


def _pose(elem):
    """<pose>x y z roll pitch yaw</pose> -> list of 6 floats (zeros if missing)."""
    if elem is None or not (elem.text or "").strip():
        return [0.0] * 6
    v = [float(t) for t in elem.text.split()]
    return v + [0.0] * (6 - len(v))


def _compose(parent, child):
    """Pose of `child` (given in the parent's frame) expressed in the parent's parent frame.
    Planar (x, y, yaw) is enough for walls; z is simply added."""
    c, s = math.cos(parent[5]), math.sin(parent[5])
    return [parent[0] + c * child[0] - s * child[1],
            parent[1] + s * child[0] + c * child[1],
            parent[2] + child[2], 0.0, 0.0, parent[5] + child[5]]


def load_world_rects(world_file, offset=(0.0, 0.0), skip=_SKIP_MODELS):
    """All box (and cylinder) collisions of the world that the robot can bump into, as
    axis-aligned rectangles (xmin, ymin, xmax, ymax). Moving actors are ignored.

    offset = (ox, oy) is ADDED to every coordinate: map = world + offset. Use it when the
    map frame (SLAM/AMCL) does not have the same origin as the Gazebo world."""
    root = ET.parse(world_file).getroot()
    world = root.find("world") if root.tag == "sdf" else root
    rects, skipped = [], 0
    for model in world.findall("model"):
        name = model.get("name", "")
        if name.startswith(skip):
            continue
        mp = _pose(model.find("pose"))
        for link in model.findall("link"):
            lp = _compose(mp, _pose(link.find("pose")))
            for col in link.findall("collision"):
                geo = col.find("geometry")
                if geo is None:
                    continue
                cp = _compose(lp, _pose(col.find("pose")))
                box, cyl = geo.find("box"), geo.find("cylinder")
                if box is not None:
                    sx, sy, sz = [float(t) for t in box.find("size").text.split()]
                elif cyl is not None:
                    r = float(cyl.find("radius").text)
                    sx = sy = 2 * r
                    sz = float(cyl.find("length").text)
                else:
                    skipped += 1             # plane, mesh, sphere ... not handled
                    continue
                z_lo, z_hi = cp[2] - sz / 2, cp[2] + sz / 2
                if z_hi < 0.05 or z_lo > 0.4:   # not at robot height (floor decals, ceilings)
                    continue
                c, s = abs(math.cos(cp[5])), abs(math.sin(cp[5]))
                hx, hy = (sx * c + sy * s) / 2, (sx * s + sy * c) / 2
                rects.append((cp[0] + offset[0] - hx, cp[1] + offset[1] - hy,
                              cp[0] + offset[0] + hx, cp[1] + offset[1] + hy))
    return rects


def area_of(rects, margin=1.0):
    """Planning area [min_x, max_x, min_y, max_y] that covers all walls plus a margin."""
    return [min(r[0] for r in rects) - margin, max(r[2] for r in rects) + margin,
            min(r[1] for r in rects) - margin, max(r[3] for r in rects) + margin]


def make_map(world_file="", offset=(0.0, 0.0)):
    """(rects, rand_area): from the world file if given, else the old sim_world walls."""
    if world_file:
        rects = load_world_rects(world_file, offset)
        if not rects:
            raise ValueError(f"no walls found in {world_file}")
        return rects, area_of(rects)
    return build_rects(), list(RAND_AREA)