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
