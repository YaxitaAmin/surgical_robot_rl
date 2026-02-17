<<<<<<< HEAD
# Reinforcement Learning-Based Path Planning for Robotic Brain Surgery Simulation

## Overview

This project implements a reinforcement learning-based approach for autonomous path planning in robotic brain surgery simulation. The system uses Q-Learning and Deep Q-Network (DQN) algorithms to navigate a surgical needle through complex brain vasculature while avoiding blood vessels and reaching tumor targets.

## Features

- Q-Learning with tabular approach (3000 episodes, 98-100% success rate)
- Deep Q-Network (DQN) with neural network (7-128-128-64-6 architecture)
- KD-Tree spatial indexing for O(log n) collision detection
- Franka Panda 7-DOF robot arm simulation
- Inverse Kinematics for robot control
- Real-time vessel clearance monitoring (SAFE/CAUTION/WARNING)
- 3D visualization using PyBullet

## Project Structure
```
brain_surgery_docker/
├── Dockerfile
├── requirements.txt
├── README.md
├── final13.py              # Q-Learning implementation
├── brain_surgery_dqn.py    # DQN implementation
└── data/
    └── vessels/
        ├── skull.stl       # 3D skull model (161mm)
        └── vessels.stl     # Blood vessel network (49,908 vertices)
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- PyBullet
- Trimesh
- PyTorch (for DQN version)

## Installation

### Option 1: Using Docker (Recommended)

Build the Docker image:
```
docker build -t brain-surgery-sim .
```

Run Q-Learning version:
```
xhost +local:docker
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix brain-surgery-sim python3 final13.py
```

Run DQN version:
```
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix brain-surgery-sim python3 brain_surgery_dqn.py
```

### Option 2: Local Installation

Create virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Run:
```
python3 final13.py
```

## Algorithm Details

### Q-Learning Parameters
- Learning rate (alpha): 0.15
- Discount factor (gamma): 0.95
- Exploration rate (epsilon): 1.0 to 0.01
- Episodes: 3000
- Grid resolution: 2mm

### DQN Parameters
- Network architecture: 7 inputs - 128 - 128 - 64 - 6 outputs
- Experience replay buffer: 10,000
- Batch size: 64
- Target network update: Every 100 episodes

### Reward Function
- Tumor reached: +100
- Vessel collision: -100
- Progress reward: 2 x (old_distance - new_distance) - 0.5
- Timeout (100 steps): -50

### Action Space
- 6 discrete actions: +X, -X, +Y, -Y, +Z, -Z
- Step size: 2mm

## Results

| Metric | Q-Learning | DQN |
|--------|------------|-----|
| Training Success | 99.6% | 98.8% |
| Generalization | 80-100% | 100% |
| Episodes | 3000 | 3000 |

## Usage

1. Run the simulation
2. Press ENTER to start surgery
3. Watch Phase 1: Robot approaches entry point
4. Press ENTER to insert needle
5. Watch Phase 2: Needle follows learned path with real-time clearance monitoring
6. Press ENTER to retract
7. Watch Phase 3: Needle retracts along same path

## Files Description

### final13.py
Main Q-Learning implementation with Franka Panda robot arm. Uses tabular Q-Learning to learn safe paths through vessel network.

### brain_surgery_dqn.py
Deep Q-Network implementation using PyTorch. Uses neural network for function approximation with experience replay and target network.

## Technical Components

### KD-Tree
Binary space partitioning structure for fast nearest neighbor queries. Reduces collision detection from O(n) to O(log n) for 49,908 vessel vertices.

### Inverse Kinematics
Converts Cartesian needle tip positions (x, y, z) to robot joint angles (theta1 to theta7) using PyBullet's IK solver.

### PyBullet Visualization
- Transparent skull with visible blood vessels
- Franka Panda robot arm with attached needle
- Path visualization (yellow for learned path, green for straight path)
- Entry point and tumor markers

## Safety Margins

- SAFE: Clearance greater than 6mm (green)
- CAUTION: Clearance 4-6mm (yellow)
- WARNING: Clearance less than 4mm (red)
- COLLISION: Clearance less than 4mm triggers episode termination

## Author

Yaxita

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529-533.
3. PyBullet Documentation: https://pybullet.org
4. Franka Emika Panda: https://www.franka.de

## License

This project is for educational purposes.
=======
# 🧠 Reinforcement Learning-Based Path Planning for Robotic Brain Surgery Simulation

> Autonomous surgical needle path planning using Q-Learning and DQN in a PyBullet physics simulation with a Franka Panda robotic arm.

**University of Maryland, College Park** | Yaxita Amin & K. Manasanjani

---

## 📖 Overview

This project presents a reinforcement learning approach for autonomous path planning in robotic brain surgery simulation. An AI agent learns to navigate a surgical needle through complex 3D brain vasculature — avoiding blood vessels while reaching tumor targets — using Q-Learning and Deep Q-Network (DQN) algorithms.

Key highlights:
- **98–100% training success rate** with tabular Q-Learning over 3,000 episodes
- **80–100% generalization** to unseen tumor targets
- **0% vessel collision rate** vs. 40% for straight-line approaches
- **≥4mm safety margin** guaranteed from all blood vessels
- Full integration with a Franka Panda robotic arm via Inverse Kinematics

---

## 🎥 Demo Videos

| Without RL Path Planning | With RL Path Planning |
|:---:|:---:|
| [![Without Needle](https://img.shields.io/badge/▶_Watch-Without_Path-red?style=for-the-badge)](link-to-video-1) | [![With Needle](https://img.shields.io/badge/▶_Watch-With_RL_Path-green?style=for-the-badge)](link-to-video-2) |
| Straight-line insertion — 40% vessel collision rate | Q-Learning curved path — 0% vessel collision rate |

> 📌 Replace the links above with your actual video URLs (YouTube, Google Drive, etc.)

---

## 🏗️ System Architecture

```
STL Models (Skull, Vessels)
        ↓
Mesh Preprocessing (Trimesh, 2mm voxelize)
        ↓
KDTree & Clearance Map
        ↓
Collision Marking (unsafe < 4mm)
        ↓
3D Grid Environment (2mm voxels)
        ↓
Q-Learning Agent — Action Space: {±X, ±Y, ±Z}
        ↓
Reward Module (+100 success, -100 collision)
        ↓ ↺ experience
Policy Extraction (Greedy)
        ↓
Path Smoothing (cubic interpolation)
        ↓
Safety Checker (≥ 4mm clearance)
        ↓
Franka Panda Execution (Inverse Kinematics)
        ↓
Final Safe Trajectory
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Ubuntu (recommended) or macOS

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/rl-brain-surgery-path-planning.git
cd rl-brain-surgery-path-planning

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
pybullet
trimesh
scipy
numpy
torch
matplotlib
```

---

## 🚀 Usage

### Run Q-Learning (Path Planning + PyBullet Simulation)

```bash
python final13.py
```

### Run DQN Agent

```bash
python brain_surgery_dqn.py
```

---

## 🧪 Results

### Training Performance

| Episode | Q-Learning | DQN    | Epsilon |
|---------|------------|--------|---------|
| 500     | 84.8%      | 95.2%  | ~0.22   |
| 1000    | 98.2%      | 98.0%  | ~0.05   |
| 2000    | 99.6%      | 97.2%  | ~0.01   |
| 3000    | 99.6%      | 98.8%  | 0.01    |

### Q-Learning vs DQN

| Metric           | Q-Learning    | DQN         |
|------------------|---------------|-------------|
| Training Time    | ~2 min        | ~15 min     |
| Training Success | 94.6%         | 89.2%       |
| Test Success     | 80%           | 75%         |
| Path Quality     | 1.1×          | 1.15×       |
| Memory Usage     | 4,003 states  | 128KB model |
| Inference Time   | <1ms          | ~5ms        |

### Path Planning Method Comparison

| Metric          | Dijkstra/A* | RRT     | Potential Fields | Q-Learning  |
|-----------------|-------------|---------|------------------|-------------|
| Computation     | 30–35s      | 5–10s   | 2–5s             | ~2 min train|
| Path Quality    | 1.0–1.3×    | 1.4–1.6×| 1.2–1.4×         | ~1.1×       |
| Success Rate    | 60–65%      | 70%     | 65–70%           | **98–100%** |
| Vessel Safety   | Occasional  | Generally| >4mm*           | **Always ≥4mm** |
| Reproducibility | High        | Low     | Medium           | High        |
| Learning        | None        | None    | None             | **Yes**     |

---

## 🤖 RL Formulation

### State Space
3D voxel grid position (2mm resolution) relative to tumor location.

### Action Space
6 discrete moves: `{+X, −X, +Y, −Y, +Z, −Z}` (2mm per step)

### Reward Function
```
R(s, a, s') = R_goal + R_collision + R_timeout + R_shaping
```
- `R_goal = +100` — reaching the tumor
- `R_collision = -100` — vessel proximity < 4mm
- `R_timeout = -50` — exceeding 100 steps
- `R_shaping = Δd` — distance reduction to target

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate α | 0.15 |
| Discount factor γ | 0.95 |
| Epsilon (start → end) | 1.0 → 0.01 |
| Epsilon decay | 0.997 |
| Max steps/episode | 100 |
| Training episodes | 3,000 |

---

## 🦾 Robotic Arm Integration

- **Robot**: Franka Emika Panda (7 DOF, simulated in PyBullet)
- **IK Solver**: PyBullet damped least-squares with joint limit handling
- **IK success rate**: 98%
- **End-effector accuracy**: <0.5mm positioning error
- **Average execution time**: 8.5 seconds per path

---

## 📁 Project Structure

```
brain_surgery_docker/
├── data/                       # STL models & brain vasculature data (~7.5MB)
├── final13.py                  # Q-Learning main script (path planning + simulation)
├── brain_surgery_dqn.py        # Deep Q-Network implementation
├── Dockerfile                  # Docker container configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🐳 Docker

A `Dockerfile` is included for containerized, reproducible execution.

### Build the Image

```bash
docker build -t brain-surgery-rl .
```

### Run the Container

```bash
docker run --rm brain-surgery-rl
```

> **Note:** PyBullet GUI visualization requires passing through a display. On Linux, use:
> ```bash
> docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix brain-surgery-rl
> ```

---

## ⚠️ Limitations

- Static environment (no tissue deformation modeling)
- Discrete action space limits path smoothness
- Single-needle, straight-segment paths only
- Simulated environment differs from real surgical conditions

---

## 🔭 Future Work

- Extend to continuous state/action spaces using actor-critic methods
- Incorporate curved needle steering for challenging targets
- Multi-objective optimization (path length, clearance, energy)
- Dynamic replanning with real-time MRI/CT intraoperative feedback
- Sim-to-real transfer for physical Franka Panda deployment
- Training on diverse anatomical models for patient-agnostic planning

---

## 📄 Citation

```bibtex
@article{amin2024rl_surgery,
  title={Reinforcement Learning-Based Path Planning for Robotic Brain Surgery Simulation},
  author={Amin, Yaxita and Manasanjani, K.},
  institution={University of Maryland, College Park},
  year={2024}
}
```

---

## 🙏 Acknowledgments

We thank **Dr. Jerry Wu** for guidance throughout this project, and teaching assistants **Siddhant** and **Aswin** for valuable feedback during development.

---

## 📜 License

This project is intended for educational and pre-operative planning research purposes only. Not for clinical use.
>>>>>>> fd9661594802d7ee827e5105a317ead44253a9bd
