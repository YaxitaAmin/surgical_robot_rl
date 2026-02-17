#!/usr/bin/env python3
"""
================================================================================
DQN BRAIN SURGERY SIMULATION
Deep Q-Network with Neural Network (instead of Q-Table)
================================================================================
"""

import numpy as np
import trimesh
from scipy.spatial import KDTree
import pybullet as p
import pybullet_data
import time
import math
import random
from collections import deque

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    print("✓ PyTorch loaded successfully!")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except ImportError:
    print("✗ PyTorch not found! Install with: pip install torch")
    exit()


# =============================================================================
# NEURAL NETWORK
# =============================================================================
class DQNNetwork(nn.Module):
    """Deep Q-Network - 3 hidden layers"""
    
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# =============================================================================
# DQN AGENT
# =============================================================================
class DQNAgent:
    """DQN Agent with Experience Replay and Target Network"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Experience Replay Memory
        self.memory = deque(maxlen=10000)
        
        # Device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return q_values.argmax().item()
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (from target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.criterion(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


# =============================================================================
# ENVIRONMENT
# =============================================================================
class BrainEnv:
    """Brain Surgery Environment with Continuous State Space"""
    
    def __init__(self, vessel_tree, entry, target):
        self.vessel_tree = vessel_tree
        self.entry = np.array(entry, dtype=float)
        self.target = np.array(target, dtype=float)
        self.step_size = 2.0  # 2mm steps
        self.safety_margin = 4.0  # 4mm from vessels
        
        # 6 actions: ±X, ±Y, ±Z
        self.actions = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]) * self.step_size
        
        self.reset()
    
    def reset(self, entry=None, target=None):
        """Reset environment"""
        if entry is not None:
            self.entry = np.array(entry)
        if target is not None:
            self.target = np.array(target)
        
        self.pos = self.entry.copy()
        self.path = [self.pos.copy()]
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self):
        """
        Continuous state representation (7 features):
        - Relative position to target (3)
        - Distance to target (1)
        - Clearance from vessels (1)
        - Normalized step count (1)
        - Distance traveled (1)
        """
        rel_pos = (self.target - self.pos) / 100.0  # Normalize
        dist_to_target = np.linalg.norm(self.target - self.pos) / 100.0
        clearance = min(self.vessel_tree.query(self.pos)[0], 20.0) / 20.0
        step_norm = self.steps / 100.0
        dist_traveled = len(self.path) * self.step_size / 100.0
        
        state = np.concatenate([
            rel_pos,                    # 3 features
            [dist_to_target],           # 1 feature
            [clearance],                # 1 feature
            [step_norm],                # 1 feature
            [dist_traveled]             # 1 feature
        ])
        
        return state  # 7 features total
    
    def step(self, action):
        """Take action, return (state, reward, done, info)"""
        self.steps += 1
        
        old_dist = np.linalg.norm(self.pos - self.target)
        new_pos = self.pos + self.actions[action]
        
        # Check vessel clearance
        clearance = self.vessel_tree.query(new_pos)[0]
        
        # Calculate reward
        if clearance < self.safety_margin:
            # Collision!
            return self._get_state(), -100, True, 'collision'
        
        self.pos = new_pos
        self.path.append(self.pos.copy())
        new_dist = np.linalg.norm(self.pos - self.target)
        
        # Reached tumor?
        if new_dist < self.step_size * 1.5:
            return self._get_state(), +100, True, 'success'
        
        # Timeout?
        if self.steps >= 100:
            return self._get_state(), -50, True, 'timeout'
        
        # Normal step reward
        reward = (old_dist - new_dist) * 2 - 0.5
        
        # Bonus for good clearance
        if clearance > 8.0:
            reward += 0.5
        
        return self._get_state(), reward, False, 'ongoing'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def find_targets(vessel_tree, skull_top, n=25):
    """Find safe tumor locations"""
    targets = []
    for _ in range(n * 100):
        if len(targets) >= n:
            break
        tumor = np.array([
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20),
            np.random.uniform(skull_top - 50, skull_top - 15)
        ])
        if vessel_tree.query(tumor)[0] < 8:
            continue
        entry = np.array([tumor[0], tumor[1], skull_top - 3])
        pts = [entry + t * (tumor - entry) for t in np.linspace(0, 1, 20)]
        if min(vessel_tree.query(pt)[0] for pt in pts) >= 3:
            targets.append((entry, tumor))
    return targets


def straight_path(entry, target, vessel_tree):
    """Generate straight line path"""
    pts = np.array([entry + t * (target - entry) for t in np.linspace(0, 1, 50)])
    min_vd = min(vessel_tree.query(pt)[0] for pt in pts)
    return pts, min_vd, min_vd >= 4.0


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("   DQN BRAIN SURGERY SIMULATION")
    print("   Deep Q-Network with Neural Network")
    print("=" * 70)
    
    # =========================================================================
    # [1] LOAD ANATOMY
    # =========================================================================
    print("\n[1] LOADING BRAIN ANATOMY...")
    skull = trimesh.load("data/vessels/skull.stl")
    vessels = trimesh.load("data/vessels/vessels.stl")
    center = skull.centroid
    skull.vertices -= center
    vessels.vertices -= center
    
    bounds = skull.bounds
    skull_min_z, skull_max_z = bounds[0, 2], bounds[1, 2]
    
    skull.export("/tmp/skull.obj")
    vessels.export("/tmp/vessels.obj")
    vessel_tree = KDTree(vessels.vertices)
    print(f"    Skull: {skull_max_z - skull_min_z:.0f}mm, Vessels: {len(vessels.vertices)}")
    
    # =========================================================================
    # [2] GENERATE TARGETS
    # =========================================================================
    print("\n[2] GENERATING TARGETS...")
    all_targets = find_targets(vessel_tree, skull_max_z, 25)
    train_targets, test_targets = all_targets[:20], all_targets[20:]
    print(f"    Train: {len(train_targets)}, Test: {len(test_targets)}")
    
    # =========================================================================
    # [3] TRAIN DQN
    # =========================================================================
    print("\n[3] TRAINING DQN...")
    print("    Network: 7 inputs → 128 → 128 → 64 → 6 outputs")
    print("    Experience Replay: 10,000 memory")
    print("    Batch Size: 64")
    print()
    
    # Initialize
    STATE_SIZE = 7   # 7 continuous features
    ACTION_SIZE = 6  # 6 directions
    
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    env = BrainEnv(vessel_tree, *train_targets[0])
    
    # Training loop
    episodes = 3000
    history = []
    losses = []
    
    for ep in range(episodes):
        # Random target from training set
        entry, target = train_targets[ep % len(train_targets)]
        state = env.reset(entry, target)
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train on batch
            loss = agent.replay()
            if loss > 0:
                losses.append(loss)
            
            state = next_state
            total_reward += reward
        
        history.append(info == 'success')
        
        # Update target network every 100 episodes
        if (ep + 1) % 100 == 0:
            agent.update_target_network()
        
        # Progress
        if (ep + 1) % 500 == 0:
            success_rate = np.mean(history[-500:]) * 100
            avg_loss = np.mean(losses[-1000:]) if losses else 0
            print(f"    Ep {ep+1:4d} | Success: {success_rate:5.1f}% | "
                  f"Loss: {avg_loss:.4f} | ε: {agent.epsilon:.3f}")
    
    train_sr = np.mean(history[-500:]) * 100
    
    # =========================================================================
    # [4] TEST GENERALIZATION
    # =========================================================================
    print("\n[4] TESTING ON UNSEEN TARGETS...")
    gen_ok = 0
    for entry, target in test_targets:
        state = env.reset(entry, target)
        done = False
        while not done:
            action = agent.act(state, training=False)
            state, _, done, info = env.step(action)
        if info == 'success':
            gen_ok += 1
    gen_rate = gen_ok / len(test_targets) * 100
    print(f"    Result: {gen_rate:.0f}% ({gen_ok}/{len(test_targets)})")
    
    # =========================================================================
    # [5] PATH COMPARISON
    # =========================================================================
    print("\n[5] PATH COMPARISON...")
    vis_entry, vis_target = train_targets[0]
    direct_dist = np.linalg.norm(vis_target - vis_entry)
    sp_pts, sp_vd, sp_safe = straight_path(vis_entry, vis_target, vessel_tree)
    
    state = env.reset(vis_entry, vis_target)
    done = False
    while not done:
        action = agent.act(state, training=False)
        state, _, done, _ = env.step(action)
    cp_pts = np.array(env.path)
    cp_len = np.sum(np.linalg.norm(np.diff(cp_pts, axis=0), axis=1))
    cp_vd = min(vessel_tree.query(pt)[0] for pt in cp_pts)
    
    print(f"    Direct: {direct_dist:.1f}mm")
    print(f"    Straight: {sp_vd:.1f}mm clearance - {'SAFE' if sp_safe else 'UNSAFE'}")
    print(f"    DQN Path: {cp_vd:.1f}mm clearance, {cp_len:.1f}mm length")
    
    # =========================================================================
    # [6] VISUALIZATION
    # =========================================================================
    print("\n[6] VISUALIZATION...")
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    SKULL_SCALE = 0.0015
    TABLE_HEIGHT = 0.35
    
    ROBOT_POS = [0, 0, 0]
    SKULL_X = 0.4
    SKULL_Y = 0.35
    SKULL_Z = TABLE_HEIGHT + 0.12
    SKULL_POS = np.array([SKULL_X, SKULL_Y, SKULL_Z])
    
    def to_world(pt):
        return SKULL_POS + np.array(pt) * SKULL_SCALE
    
    entry_world = to_world(vis_entry)
    target_world = to_world(vis_target)
    
    # Scene setup
    p.loadURDF("plane.urdf")
    
    # Table
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, TABLE_HEIGHT/2],
                                     rgbaColor=[0.3, 0.5, 0.35, 1])
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, TABLE_HEIGHT/2])
    p.createMultiBody(0, table_col, table_vis, [SKULL_X, SKULL_Y, TABLE_HEIGHT/2])
    
    p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_BOX, halfExtents=[0.14, 0.14, 0.008],
                           rgbaColor=[0.25, 0.55, 0.35, 1]),
        [SKULL_X, SKULL_Y, TABLE_HEIGHT + 0.008])
    
    # Skull and vessels
    skull_vis = p.createVisualShape(p.GEOM_MESH, fileName="/tmp/skull.obj",
                                     meshScale=[SKULL_SCALE]*3,
                                     rgbaColor=[0.95, 0.9, 0.85, 0.35])
    p.createMultiBody(0, -1, skull_vis, SKULL_POS.tolist())
    
    vessel_vis = p.createVisualShape(p.GEOM_MESH, fileName="/tmp/vessels.obj",
                                      meshScale=[SKULL_SCALE]*3,
                                      rgbaColor=[0.85, 0.12, 0.12, 0.9])
    p.createMultiBody(0, -1, vessel_vis, SKULL_POS.tolist())
    
    # Markers
    p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_SPHERE, radius=0.006, rgbaColor=[0.2, 0.5, 1, 1]),
        entry_world.tolist())
    p.addUserDebugText("ENTRY", (entry_world + [0, 0, 0.025]).tolist(), [0, 0.4, 1], 1.0)
    
    p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[0.2, 1, 0.3, 1]),
        target_world.tolist())
    p.addUserDebugText("TUMOR", (target_world + [0.02, 0, 0]).tolist(), [0, 0.7, 0.2], 1.0)
    
    # Paths
    sp_w = np.array([to_world(pt) for pt in sp_pts])
    cp_w = np.array([to_world(pt) for pt in cp_pts])
    
    sp_color = [0.3, 0.9, 0.3] if sp_safe else [1, 0.5, 0.1]
    for i in range(len(sp_w)-1):
        p.addUserDebugLine(sp_w[i].tolist(), sp_w[i+1].tolist(), sp_color, 2)
    for i in range(len(cp_w)-1):
        p.addUserDebugLine(cp_w[i].tolist(), cp_w[i+1].tolist(), [1, 0.9, 0.1], 3)
    
    # Robot
    print("    Loading Franka Panda...")
    panda = p.loadURDF("franka_panda/panda.urdf", ROBOT_POS, useFixedBase=True)
    
    EE_LINK = 11
    NUM_JOINTS = 7
    
    joint_limits_low = []
    joint_limits_high = []
    for i in range(NUM_JOINTS):
        info = p.getJointInfo(panda, i)
        joint_limits_low.append(info[8])
        joint_limits_high.append(info[9])
    
    start_pos = entry_world + np.array([0, 0, 0.15])
    desired_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    def solve_ik(target_pos, target_orn):
        return p.calculateInverseKinematics(
            panda, EE_LINK, target_pos.tolist(), target_orn,
            lowerLimits=joint_limits_low, upperLimits=joint_limits_high,
            jointRanges=[7]*NUM_JOINTS, restPoses=[0, -0.5, 0, -2, 0, 2, 0.785],
            maxNumIterations=100, residualThreshold=1e-5
        )[:NUM_JOINTS]
    
    def set_joints(joint_poses):
        for i in range(NUM_JOINTS):
            p.resetJointState(panda, i, joint_poses[i])
        p.resetJointState(panda, 9, 0.02)
        p.resetJointState(panda, 10, 0.02)
    
    # Move to start
    start_joints = solve_ik(start_pos, desired_orn)
    set_joints(start_joints)
    
    # Needle
    NEEDLE_LEN = 0.10
    needle_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.003, length=NEEDLE_LEN,
                                      rgbaColor=[0.8, 0.8, 0.85, 1])
    needle_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.003, height=NEEDLE_LEN)
    needle_body = p.createMultiBody(0.01, needle_col, needle_vis, [0, 0, 0])
    
    needle_constraint = p.createConstraint(
        panda, EE_LINK, needle_body, -1, p.JOINT_FIXED, [0, 0, 1],
        [0, 0, 0.05], [0, 0, NEEDLE_LEN/2]
    )
    
    tip_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[1, 0.2, 0.2, 1])
    tip_body = p.createMultiBody(0, -1, tip_vis, [0, 0, 0])
    
    def update_tip():
        needle_state = p.getBasePositionAndOrientation(needle_body)
        pos = np.array(needle_state[0])
        orn = needle_state[1]
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        tip_pos = pos - rot @ np.array([0, 0, NEEDLE_LEN/2])
        p.resetBasePositionAndOrientation(tip_body, tip_pos.tolist(), [0, 0, 0, 1])
        return tip_pos
    
    for _ in range(50):
        p.stepSimulation()
        update_tip()
    
    # Camera
    p.resetDebugVisualizerCamera(0.8, 35, -25, [0.3, 0.25, 0.4])
    
    print("\n" + "=" * 70)
    print("DQN READY - Neural Network trained!")
    print("=" * 70)
    
    input("\n>>> Press ENTER to start surgery...")
    
    # =========================================================================
    # SURGERY ANIMATION (same as before)
    # =========================================================================
    print("\n  Phase 1: Approaching...")
    hover_pos = entry_world + np.array([0, 0, 0.02])
    
    for i in range(50):
        t = i / 50
        current_tip = start_pos + t * (hover_pos - start_pos)
        ee_target = current_tip + np.array([0, 0, NEEDLE_LEN + 0.03])
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        for _ in range(5):
            p.stepSimulation()
        update_tip()
        time.sleep(0.05)
    
    print("    ✓ At entry point!")
    input("\n>>> Press ENTER to insert needle...")
    
    print("\n  Phase 2: Inserting (DQN path)...")
    
    # Penetrate
    for i in range(15):
        t = i / 15
        current_tip = hover_pos + t * (entry_world - hover_pos)
        ee_target = current_tip + np.array([0, 0, NEEDLE_LEN + 0.03])
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        for _ in range(3):
            p.stepSimulation()
        update_tip()
        time.sleep(0.08)
    
    # Follow DQN path
    for i, brain_pt in enumerate(cp_pts):
        tip_target = to_world(brain_pt)
        ee_target = tip_target + np.array([0, 0, NEEDLE_LEN + 0.03])
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        for _ in range(3):
            p.stepSimulation()
        update_tip()
        
        clearance = vessel_tree.query(brain_pt)[0]
        depth_mm = np.linalg.norm(brain_pt - vis_entry)
        pct = (i + 1) / len(cp_pts) * 100
        
        status = "✓ SAFE" if clearance >= 6 else "⚡ CAUTION" if clearance >= 4 else "⚠️ WARNING"
        print(f"\r    Depth: {depth_mm:5.1f}mm | Clearance: {clearance:4.1f}mm | {status} | {pct:5.1f}%   ", end="")
        
        time.sleep(0.15)
    
    print("\n\n  ✓ TUMOR REACHED!")
    input("\n>>> Press ENTER to retract...")
    
    # Retract
    print("\n  Phase 3: Retracting...")
    for i, brain_pt in enumerate(cp_pts[::-1]):
        tip_target = to_world(brain_pt)
        ee_target = tip_target + np.array([0, 0, NEEDLE_LEN + 0.03])
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        for _ in range(3):
            p.stepSimulation()
        update_tip()
        time.sleep(0.1)
        pct = (i + 1) / len(cp_pts) * 100
        print(f"\r    Retracting: {pct:5.1f}%", end="")
    
    print("\n\n  ✓ Surgery complete!")
    
    # Results
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              DQN SURGERY RESULTS                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Network:        7 → 128 → 128 → 64 → 6                      ║
    ║  Training:       {train_sr:5.1f}%                                    ║
    ║  Generalization: {gen_rate:5.1f}% ({gen_ok}/{len(test_targets)})                                ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Straight: {sp_vd:5.1f}mm clearance {'✓' if sp_safe else '✗'}                           ║
    ║  DQN Path: {cp_vd:5.1f}mm clearance, {cp_len:5.1f}mm path                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    input(">>> Press ENTER to exit...")
    p.disconnect()


if __name__ == "__main__":
    main()
