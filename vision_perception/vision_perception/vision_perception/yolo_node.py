import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


# ----------------------------------------------------------------------
# 1. Tracking + activity logic (no ROS code inside, easy to test)
#    Own tracker: stable IDs, motion prediction, "coasting" through
#    missed detections, hysteresis so labels do not flicker.
# ----------------------------------------------------------------------
def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class PersonState:
    """Everything we remember about ONE tracked person."""

    def __init__(self, pid, box, now):
        self.id = pid
        self.box = box                  # last real detection (x1,y1,x2,y2)
        self.sx = self.sy = None        # smoothed centre
        self.history = deque()          # recent (time, sx, sy)
        self.velocity = 0.0             # px/s
        self.vec = (0.0, 0.0)           # px/s (x, y) used for prediction
        self.activity = "Staying"
        self.stay_start = None
        self.goal_sent = False
        self.last_seen = now
        self.hits = 0
        self.visible = True             # detected in the current frame


class ActivityTracker:
    def __init__(self, walk_threshold=20.0, run_threshold=150.0,
                 window=1.5, min_dt=0.4, stay_seconds=5.0,
                 max_age=8.0, min_hits=2, hysteresis=0.7,
                 smooth=0.6, iou_min=0.05, dist_gate=1.5,
                 min_gate_px=80.0, max_speed=400.0, max_gate_px=600.0,
                 reset_gap=6.0):
        self.walk_threshold = walk_threshold
        self.run_threshold = run_threshold
        self.window = window
        self.min_dt = min_dt
        self.stay_seconds = stay_seconds
        self.max_age = max_age          # keep a lost person this long
        self.min_hits = min_hits        # detections before a track is real
        self.hysteresis = hysteresis
        self.smooth = smooth
        self.iou_min = iou_min
        self.dist_gate = dist_gate      # in box heights
        self.min_gate_px = min_gate_px  # never gate tighter than this
        self.max_speed = max_speed      # px/s a person can plausibly move
        self.max_gate_px = max_gate_px  # never match farther than this
        self.reset_gap = reset_gap      # unseen this long -> forget old speed
        self.frame_period = 0.0         # smoothed seconds between frames
        self.last_step = None
        self.people = {}
        self.next_id = 1

    # ---- activity with hysteresis (no flicker around thresholds) ----
    def classify(self, v, prev):
        walk_t = self.walk_threshold if prev == "Staying" \
            else self.walk_threshold * self.hysteresis
        run_t = self.run_threshold if prev != "Running" \
            else self.run_threshold * self.hysteresis
        if v < walk_t:
            return "Staying"
        if v < run_t:
            return "Walking"
        return "Running"

    def _predict(self, st, now):
        dt = min(now - st.last_seen, 0.5)
        dx, dy = st.vec[0] * dt, st.vec[1] * dt
        x1, y1, x2, y2 = st.box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _observe(self, st, box, now):
        gap = now - st.last_seen
        if st.hits > 0 and gap > max(self.reset_gap, 4.0 * self.frame_period):
            # person was out of view for a while (e.g. behind a wall):
            # old positions say nothing about their speed now
            st.history.clear()
            st.sx = st.sy = None
            st.vec = (0.0, 0.0)
            st.stay_start = None
            st.goal_sent = False
        st.box = box
        st.last_seen = now
        st.visible = True
        st.hits += 1
        cx, cy = _center(box)
        if st.sx is None:
            st.sx, st.sy = cx, cy
        else:
            a = self.smooth
            st.sx += a * (cx - st.sx)
            st.sy += a * (cy - st.sy)
        st.history.append((now, st.sx, st.sy))
        # keep ~window seconds of history, but ALWAYS at least 2 samples,
        # so speed still works when the node runs at under 1 frame/s
        while len(st.history) > 2 and now - st.history[0][0] > self.window:
            st.history.popleft()

        t0, x0, y0 = st.history[0]
        dt = now - t0
        if dt >= self.min_dt:
            st.velocity = math.hypot(st.sx - x0, st.sy - y0) / dt
            st.vec = ((st.sx - x0) / dt, (st.sy - y0) / dt)
        st.activity = self.classify(st.velocity, st.activity)

        if st.activity == "Staying":
            if st.stay_start is None and dt >= self.min_dt:
                st.stay_start = now
        else:
            st.stay_start = None
            st.goal_sent = False

    def step(self, dets, now):
        """dets: list of (x1,y1,x2,y2). Returns confirmed PersonStates
        (visible ones were detected now, the others are coasting)."""
        for st in self.people.values():
            st.visible = False
        if self.last_step is not None:
            dtf = now - self.last_step
            self.frame_period = dtf if self.frame_period == 0.0 \
                else 0.7 * self.frame_period + 0.3 * dtf
        self.last_step = now

        pairs = []
        for pid, st in self.people.items():
            pb = self._predict(st, now)
            pcx, pcy = _center(pb)
            h = max(pb[3] - pb[1], 1.0)
            # allowed jump grows with box size AND with time since last seen
            gate = max(self.dist_gate * h, self.min_gate_px) \
                + self.max_speed * (now - st.last_seen)
            gate = min(gate, self.max_gate_px)
            for j, d in enumerate(dets):
                dcx, dcy = _center(d)
                dist = math.hypot(dcx - pcx, dcy - pcy)
                ov = _iou(pb, d)
                if ov >= self.iou_min or dist <= gate:
                    pairs.append((dist / gate - ov, pid, j))
        pairs.sort()

        used_t, used_d = set(), set()
        for _, pid, j in pairs:
            if pid in used_t or j in used_d:
                continue
            used_t.add(pid)
            used_d.add(j)
            self._observe(self.people[pid], tuple(dets[j]), now)

        for j, d in enumerate(dets):
            if j not in used_d:
                st = PersonState(self.next_id, tuple(d), now)
                self.next_id += 1
                self.people[st.id] = st
                self._observe(st, tuple(d), now)
                st.hits = 1

        for pid in [p for p, s in self.people.items()
                    if now - s.last_seen > self.max_age]:
            del self.people[pid]

        return list(self.people.values())

    def stay_ready(self, st, now):
        return (st.visible and st.hits >= self.min_hits
                and st.stay_start is not None
                and now - st.stay_start >= self.stay_seconds
                and not st.goal_sent)


# ----------------------------------------------------------------------
# 2. Homography calibration and grid mapping - Chapter 9
# ----------------------------------------------------------------------
def load_calibration(path, offset_x=0.0, offset_y=0.0):
    """Read 4+ pixel <-> map point pairs from saved_calibration.json.
    If the file says "frame": "world", the offset is added so the points
    end up in the planner's map frame."""
    with open(path) as f:
        data = json.load(f)
    img = np.float32(data["image_points"])
    mp = np.float32(data["map_points"])
    if img.ndim != 2 or img.shape[1] != 2 or img.shape != mp.shape \
            or img.shape[0] < 4:
        raise ValueError("calibration needs 4+ matching (x, y) point pairs")
    if data.get("frame", "map") == "world":
        mp = mp + np.float32([offset_x, offset_y])
    return img, mp


def save_calibration(path, img, mp):
    with open(path, "w") as f:
        json.dump({"frame": "map",
                   "image_points": np.asarray(img).tolist(),
                   "map_points": np.asarray(mp).tolist()}, f, indent=2)


class InteractiveCalibration:
    """Click 4 floor points on the first frame, then type their real
    map coordinates in the terminal (Listing 9.2 of the guide)."""

    def __init__(self):
        self.clicked_points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicked_points) < 4:
            self.clicked_points.append([x, y])

    def run(self, frame):
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        print("\nClick 4 FLOOR points whose map coordinates you know "
              "(e.g. where actors' feet stand).")
        print("Press 'r' to restart the clicks, Enter or Space when 4 are set.")
        while True:
            show = frame.copy()
            for i, (x, y) in enumerate(self.clicked_points):
                cv2.circle(show, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(show, str(i + 1), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Calibration", show)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r'):
                self.clicked_points = []
            if len(self.clicked_points) == 4 and key in (13, 32):
                break
        cv2.destroyWindow("Calibration")

        map_points = []
        for i in range(4):
            mx = float(input(f"Point {i + 1} map X: "))
            my = float(input(f"Point {i + 1} map Y: "))
            map_points.append([mx, my])
        return np.float32(self.clicked_points), np.float32(map_points)


class GridMapper:
    """Foot pixel -> map coordinate (homography) -> nearest grid point."""

    def __init__(self, H, map_pts, step=0.5):
        self.H = H
        self.step = step
        self.x_min = float(map_pts[:, 0].min())
        self.y_min = float(map_pts[:, 1].min())
        # same grid as np.arange(min, max + step, step) in Listing 9.4
        self.nx = len(np.arange(self.x_min, float(map_pts[:, 0].max()) + step, step))
        self.ny = len(np.arange(self.y_min, float(map_pts[:, 1].max()) + step, step))

    def to_map(self, u, v):
        pt = np.float32([[[u, v]]])
        dst = cv2.perspectiveTransform(pt, self.H)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def foot_to_grid(self, u, v):
        mx, my = self.to_map(u, v)
        # nearest grid node on a regular grid = round per axis, kept inside
        ix = int(min(max(math.floor((mx - self.x_min) / self.step + 0.5), 0),
                     self.nx - 1))
        iy = int(min(max(math.floor((my - self.y_min) / self.step + 0.5), 0),
                     self.ny - 1))
        gx = self.x_min + ix * self.step
        gy = self.y_min + iy * self.step
        return gx, gy, mx, my          # snapped goal + raw coordinate


COLORS = {
    "Staying": (0, 255, 0),
    "Walking": (0, 200, 255),
    "Running": (0, 0, 255),
}


# ----------------------------------------------------------------------
# 3. The ROS 2 node
# ----------------------------------------------------------------------
class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Parameters (change with --ros-args -p name:=value)
        self.declare_parameter('image_topic', '/surveillance_camera/image_raw')
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('imgsz', 416)
        self.declare_parameter('walk_threshold', 20.0)
        self.declare_parameter('run_threshold', 150.0)
        self.declare_parameter('stay_seconds', 5.0)
        self.declare_parameter('goal_cooldown', 15.0)
        self.declare_parameter('process_interval', 0.0)   # 0 = every frame
        self.declare_parameter('max_age', 8.0)   # seconds a lost person is remembered
        self.declare_parameter('min_hits', 2)
        self.declare_parameter('show_window', True)
        # homography
        self.declare_parameter('use_homography', True)
        self.declare_parameter('calibration_file',
                               os.path.expanduser('~/saved_calibration.json'))
        self.declare_parameter('grid_step', 0.5)
        self.declare_parameter('map_offset_x', 0.0)
        self.declare_parameter('map_offset_y', 0.0)
        self.declare_parameter('map_bounds', [-30.0, 30.0, -30.0, 30.0])
        # used only when there is no calibration
        self.declare_parameter('target_x', -14.0)
        self.declare_parameter('target_y', 0.5)

        p = lambda name: self.get_parameter(name).value
        self.process_interval = p('process_interval')
        self.show_window = p('show_window')
        self.goal_cooldown = p('goal_cooldown')
        self.use_homography = p('use_homography')
        self.calib_path = p('calibration_file')
        self.grid_step = p('grid_step')
        self.offset_x = p('map_offset_x')
        self.offset_y = p('map_offset_y')
        self.map_bounds = list(p('map_bounds'))
        self.target_x = p('target_x')
        self.target_y = p('target_y')
        self.conf = p('conf')
        self.imgsz = p('imgsz')
        self.min_hits = p('min_hits')
        self.base_max_age = p('max_age')
        self.prev_frame_time = None

        self.tracker = ActivityTracker(
            walk_threshold=p('walk_threshold'),
            run_threshold=p('run_threshold'),
            stay_seconds=p('stay_seconds'),
            max_age=p('max_age'),
            min_hits=p('min_hits'))

        self.mapper = None
        self.calibration_done = not self.use_homography

        self.bridge = CvBridge()
        self.model = YOLO(p('model'))
        # warm-up so the first real frame is not slow
        self.model.predict(np.zeros((480, 640, 3), np.uint8),
                           imgsz=self.imgsz, verbose=False)
        self.last_process_time = 0.0
        self.last_goal_time = 0.0

        # Best-effort, depth 1 = always the NEWEST frame, never a backlog
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, p('image_topic'),
                                 self.image_callback, qos)

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.activity_pub = self.create_publisher(String, '/person_activity', 10)
        self.activity_all_pub = self.create_publisher(
            String, '/person_activity_all', 10)
        self.annotated_pub = self.create_publisher(
            CompressedImage, '/yolo/annotated/compressed', 1)

        if self.show_window:
            cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        self.get_logger().info("YOLO tracking + homography node READY")

    # ---------------- calibration (Chapter 9.3, 14.10) ----------------
    def setup_calibration(self, frame):
        self.calibration_done = True
        non_interactive = (not sys.stdin.isatty()
                           or os.environ.get('YOLO_NO_INTERACTIVE') == '1')

        img = mp = None
        use_existing = False
        if os.path.exists(self.calib_path):
            if non_interactive:
                use_existing = True
                self.get_logger().info(
                    "Non-interactive mode: loading saved calibration.")
            else:
                ans = input("\nUse existing calibration? (y/n): ").strip().lower()
                use_existing = (ans == 'y')

        if use_existing:
            try:
                img, mp = load_calibration(
                    self.calib_path, self.offset_x, self.offset_y)
            except Exception as e:
                self.get_logger().warn(f"Could not read calibration: {e}")

        if img is None and not non_interactive:
            img, mp = InteractiveCalibration().run(frame)
            save_calibration(self.calib_path, img, mp)
            self.get_logger().info(f"Calibration saved to {self.calib_path}")

        if img is None:
            self.get_logger().warn(
                "No calibration available: using the fixed target "
                f"({self.target_x}, {self.target_y}).")
            return

        H, _ = cv2.findHomography(
            img.reshape(-1, 1, 2), mp.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
        if H is None:
            self.get_logger().warn(
                "Homography failed (points collinear?). Using fixed target.")
            return
        self.mapper = GridMapper(H, mp, self.grid_step)
        self.get_logger().info("Homography ready: goals use real positions.")

    def is_goal_feasible(self, gx, gy):
        xmin, xmax, ymin, ymax = self.map_bounds
        if gx < xmin or gx > xmax:
            self.get_logger().warn(
                f"INFEASIBLE GOAL: x={gx:.2f} outside [{xmin}, {xmax}]")
            return False
        if gy < ymin or gy > ymax:
            self.get_logger().warn(
                f"INFEASIBLE GOAL: y={gy:.2f} outside [{ymin}, {ymax}]")
            return False
        return True

    # ------------------------- main callback -------------------------
    def image_callback(self, msg):
        now = time.time()
        if self.process_interval > 0 and \
                now - self.last_process_time < self.process_interval:
            return
        self.last_process_time = now

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        if not self.calibration_done:
            self.setup_calibration(frame)
            now = time.time()          # calibration may have taken a while

        # 1) detect people (class 0) - detection only, tracking is ours
        t0 = time.time()
        results = self.model.predict(
            frame, classes=[0], conf=self.conf, imgsz=self.imgsz,
            verbose=False)
        infer_ms = 1000.0 * (time.time() - t0)
        self.get_logger().info(f"inference {infer_ms:.0f} ms",
                               throttle_duration_sec=5.0)

        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                dets.append(tuple(float(v) for v in box.xyxy[0]))

        # 2) track + classify
        # If frames are slow (CPU), keep lost people for ~3 frame periods,
        # otherwise every missed detection would create a new ID.
        if self.prev_frame_time is not None:
            period = now - self.prev_frame_time
            self.tracker.max_age = max(self.base_max_age, 3.0 * period)
        self.prev_frame_time = now
        states = self.tracker.step(dets, now)

        people = []   # dicts: id, box, state, area, foot, grid, visible
        for st in states:
            if not st.visible and now - st.last_seen > 3.0:
                continue          # still remembered, just not drawn
            # lost for a moment? draw where the person should be now
            bx = st.box if st.visible else self.tracker._predict(st, now)
            x1, y1, x2, y2 = map(int, bx)
            foot = ((x1 + x2) // 2, y2)              # bottom-centre of box
            grid = None
            if self.mapper is not None and st.visible:
                grid = self.mapper.foot_to_grid(foot[0], foot[1])
            people.append({"id": st.id, "box": (x1, y1, x2, y2),
                           "state": st, "area": (x2 - x1) * (y2 - y1),
                           "foot": foot, "grid": grid,
                           "visible": st.visible})

        # status line: what the node thinks of every person (every 3 s)
        if people:
            parts = []
            for d in people:
                st = d["state"]
                stay = (now - st.stay_start) if st.stay_start else 0.0
                parts.append(
                    f"#{st.id} {st.activity} v={st.velocity:.0f} "
                    f"stay={stay:.1f}s hits={st.hits}"
                    f"{'' if st.visible else ' LOST'}"
                    f"{' sent' if st.goal_sent else ''}")
            self.get_logger().info("PEOPLE: " + " | ".join(parts),
                                   throttle_duration_sec=3.0)

        # 3) goal for every person who has been staying long enough
        for person in people:
            st = person["state"]
            if not self.tracker.stay_ready(st, now):
                continue
            if person["grid"] is not None:
                gx, gy, mx, my = person["grid"]
                if not (self.is_goal_feasible(mx, my)
                        and self.is_goal_feasible(gx, gy)):
                    st.goal_sent = True      # rejected: do not repeat warning
                    continue
            else:
                gx, gy = self.target_x, self.target_y
            if self.send_goal(st.id, gx, gy, now):
                st.goal_sent = True

        # 4) activity topics (only people seen in this frame)
        seen = [d for d in people if d["visible"]
                and d["state"].hits >= self.min_hits]
        if seen:
            main = max(seen, key=lambda d: d["area"])
            self.activity_pub.publish(String(data=main["state"].activity))
            self.activity_all_pub.publish(String(data=",".join(
                f"{d['id']}:{d['state'].activity}" for d in seen)))

        # 5) draw
        for d in people:
            x1, y1, x2, y2 = d["box"]
            st = d["state"]
            color = COLORS[st.activity]
            thick = 2 if (d["visible"] and st.hits >= self.min_hits) else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            tag = "" if d["visible"] else " (lost)"
            cv2.putText(frame, f"ID {d['id']} {st.activity}{tag}",
                        (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            cv2.putText(frame, f"v={st.velocity:.1f} px/s", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            if d["visible"]:
                cv2.circle(frame, d["foot"], 4, (255, 255, 0), -1)
            if d["grid"] is not None:
                gx, gy, mx, my = d["grid"]
                cv2.putText(frame, f"GRID ({gx:.1f}, {gy:.1f})",
                            (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"raw ({mx:.2f}, {my:.2f})",
                            (x1, y2 + 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (200, 200, 200), 1)

        # Annotated stream for the app (JPEG)
        ok, buf = cv2.imencode('.jpg', frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            out = CompressedImage()
            out.header = msg.header
            out.format = "jpeg"
            out.data = buf.tobytes()
            self.annotated_pub.publish(out)

        if self.show_window:
            cv2.imshow("YOLO Detection", frame)
            cv2.waitKey(1)

    def send_goal(self, pid, gx, gy, now):
        if now - self.last_goal_time < self.goal_cooldown:
            return False

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

        self.last_goal_time = now
        self.get_logger().info(
            f"GOAL sent for person {pid}: ({gx:.2f}, {gy:.2f})")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
