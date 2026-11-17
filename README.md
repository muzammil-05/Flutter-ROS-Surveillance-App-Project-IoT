# Autonomous ROS 2 Navigation & Vision Tracking System

An integrated robotic navigation and perception stack developed for a differential-drive robot (**TurtleBot3 Waffle**) in Gazebo. This system features a custom 3-stage planning and control pipeline coupled with a YOLO-based vision perception node for dynamic target tracking.

---

## 🏗️ System Architecture & Pipeline

Instead of relying on standard Nav2 out-of-the-box packages, this project implements a custom modular architecture:

1. **Stage 1: IA* Global Path Planner (`planner_node`)**
   - Parses the environment map directly from the Gazebo world file (`maze_world.world`).
   - Computes an optimal global path from the start pose to target destinations (including dynamic `/goal_pose` inputs).
2. **Stage 2: Path Smoother (`smoother_node`)**
   - Applies advanced path smoothing algorithms while enforcing safety clearances from wall boundaries to avoid sharp-corner collisions.
3. **Stage 3: Pure Pursuit Controller (`controller_node`)**
   - Tracks the smoothed trajectory smoothly and calculates real-time velocity commands (`/cmd_vel`) for the robot.
4. **Vision Perception Node (`vision_perception`)**
   - Utilizes YOLO (Ultralytics YOLOv8) to detect targets from camera streams and publish target coordinates to `/goal_pose`.

---

## 📂 Repository Structure

```text
src/
├── custom_navigation/          # Custom navigation stack
│   ├── custom_navigation/      # Core Python modules (IA*, smoother, controller, world map)
│   ├── launch/                 # ROS 2 launch files (custom_nav.launch.py)
│   ├── package.xml
│   └── setup.py
│
└── vision_perception/          # YOLO-based target detection and perception package
    ├── vision_perception/      # Vision nodes and model weights
    ├── package.xml
    └── setup.py

⚙️ Installation & Build
Clone the repository into your ROS 2 workspace src directory:

Bash
cd ~/root_directory/src
git clone [https://github.com/muzammil-05/Flutter-ROS-Surveillance-App-Project-IoT.git](https://github.com/muzammil-05/Flutter-ROS-Surveillance-App-Project-IoT.git)
Install required dependencies (such as Ultralytics and OpenCV):

Bash
pip3 install ultralytics opencv-python numpy
Build the workspace using colcon:

Bash
cd ~/root_directory
colcon build --symlink-install
source install/setup.bash
🚀 Usage & Launch
To launch the complete navigation stack along with simulation time and configuration parameters, run:

Bash
ros2 launch custom_navigation custom_nav.launch.py use_sim_time:=true pose_source:=tf goal_standoff:=0.8
