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

## 🎥 Demo Videos (copyrighted by yaxita and manasa)

WITHOUT ROBOT ARM

![Copy of Video Project 5 (1) (1) (1)](https://github.com/user-attachments/assets/00841dda-27bb-4a1d-95bf-f6a6a7a39508)

![Copy of Video Project 6 (1)](https://github.com/user-attachments/assets/532d5656-469f-4587-927d-4952127e97b0)

WITH ROBOT ARM AND NEEDLE

---

## 🏗️ System Architecture

![System Architecture](image.png)

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

---

## 🙏 Acknowledgments

We thank **Dr. Jerry Wu** for guidance throughout this project, and teaching assistants **Siddhant** and **Aswin** for valuable feedback during development.

---

## 📜 License

This project is intended for educational and pre-operative planning research purposes only. Not for clinical use.
