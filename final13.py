#!/usr/bin/env python3
"""
================================================================================
Q-LEARNING BRAIN SURGERY SIMULATION
ROBOT ARM MOVES DURING INSERTION
================================================================================
"""

import numpy as np
import trimesh
from scipy.spatial import KDTree
import pybullet as p
import pybullet_data
import time
import math

# =============================================================================
# Q-LEARNING
# =============================================================================
class BrainEnv:
    def __init__(self, vessel_tree, entry, target):
        self.vessel_tree = vessel_tree
        self.entry = np.array(entry, dtype=float)
        self.target = np.array(target, dtype=float)
        self.grid = 2.0
        self.safety = 4.0
        self.actions = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]) * self.grid
        self.reset()
    
    def reset(self, entry=None, target=None):
        if entry is not None: self.entry = np.array(entry)
        if target is not None: self.target = np.array(target)
        self.pos = self.entry.copy()
        self.path = [self.pos.copy()]
        self.steps = 0
        return self._state()
    
    def _state(self):
        return tuple(((self.pos - self.target) / self.grid).astype(int))
    
    def step(self, action):
        self.steps += 1
        old_d = np.linalg.norm(self.pos - self.target)
        new_pos = self.pos + self.actions[action]
        vd = self.vessel_tree.query(new_pos)[0]
        if vd < self.safety:
            return self._state(), -100, True, 'collision'
        self.pos = new_pos
        self.path.append(self.pos.copy())
        new_d = np.linalg.norm(self.pos - self.target)
        if new_d < self.grid * 1.5:
            return self._state(), 100, True, 'success'
        if self.steps >= 100:
            return self._state(), -50, True, 'timeout'
        return self._state(), (old_d - new_d) * 2 - 0.5, False, 'ongoing'


class QAgent:
    def __init__(self):
        self.q = {}
        self.lr, self.gamma, self.eps = 0.15, 0.95, 1.0
    
    def act(self, s, train=True):
        if s not in self.q: self.q[s] = np.zeros(6)
        if train and np.random.random() < self.eps: return np.random.randint(6)
        return int(np.argmax(self.q[s]))
    
    def learn(self, s, a, r, s2, done):
        if s not in self.q: self.q[s] = np.zeros(6)
        if s2 not in self.q: self.q[s2] = np.zeros(6)
        target = r if done else r + self.gamma * np.max(self.q[s2])
        self.q[s][a] += self.lr * (target - self.q[s][a])
        self.eps = max(0.01, self.eps * 0.997)


def find_targets(vessel_tree, skull_top, n=25):
    targets = []
    for _ in range(n * 100):
        if len(targets) >= n: break
        tumor = np.array([np.random.uniform(-20, 20), np.random.uniform(-20, 20),
                         np.random.uniform(skull_top - 50, skull_top - 15)])
        if vessel_tree.query(tumor)[0] < 8: continue
        entry = np.array([tumor[0], tumor[1], skull_top - 3])
        pts = [entry + t*(tumor-entry) for t in np.linspace(0, 1, 20)]
        if min(vessel_tree.query(pt)[0] for pt in pts) >= 3:
            targets.append((entry, tumor))
    return targets


def straight_path(entry, target, vessel_tree):
    pts = np.array([entry + t*(target-entry) for t in np.linspace(0, 1, 50)])
    min_vd = min(vessel_tree.query(pt)[0] for pt in pts)
    return pts, min_vd, min_vd >= 4.0


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*70)
    print("   Q-LEARNING BRAIN SURGERY SIMULATION")
    print("   ROBOT ARM MOVES DURING INSERTION")  
    print("="*70)
    
    # Load anatomy
    print("\n[1] LOADING BRAIN ANATOMY...")
    skull = trimesh.load("data/vessels/skull.stl")
    vessels = trimesh.load("data/vessels/vessels.stl")
    center = skull.centroid
    skull.vertices -= center
    vessels.vertices -= center
    
    bounds = skull.bounds
    skull_min_z, skull_max_z = bounds[0, 2], bounds[1, 2]
    skull_height_mm = skull_max_z - skull_min_z
    
    skull.export("/tmp/skull.obj")
    vessels.export("/tmp/vessels.obj")
    vessel_tree = KDTree(vessels.vertices)
    print(f"    Skull: {skull_height_mm:.0f}mm, Vessels: {len(vessels.vertices)}")
    
    # Training
    print("\n[2] GENERATING TARGETS...")
    all_targets = find_targets(vessel_tree, skull_max_z, 25)
    train_targets, test_targets = all_targets[:20], all_targets[20:]
    print(f"    Train: {len(train_targets)}, Test: {len(test_targets)}")
    
    print("\n[3] TRAINING Q-LEARNING...")
    env = BrainEnv(vessel_tree, *train_targets[0])
    agent = QAgent()
    history = []
    
    for ep in range(3000):
        e, t = train_targets[ep % len(train_targets)]
        state = env.reset(e, t)
        done = False
        while not done:
            action = agent.act(state)
            state2, reward, done, info = env.step(action)
            agent.learn(state, action, reward, state2, done)
            state = state2
        history.append(info == 'success')
        if (ep + 1) % 500 == 0:
            print(f"    Ep {ep+1} | Success: {np.mean(history[-500:])*100:.1f}%")
    
    train_sr = np.mean(history[-500:]) * 100
    
    print("\n[4] TESTING GENERALIZATION...")
    gen_ok = 0
    for e, t in test_targets:
        state = env.reset(e, t)
        done = False
        while not done:
            action = agent.act(state, False)
            state, _, done, info = env.step(action)
        if info == 'success': gen_ok += 1
    gen_rate = gen_ok / len(test_targets) * 100
    print(f"    Result: {gen_rate:.0f}% ({gen_ok}/{len(test_targets)})")
    
    print("\n[5] PATH COMPARISON...")
    vis_entry, vis_target = train_targets[0]
    direct_dist = np.linalg.norm(vis_target - vis_entry)
    sp_pts, sp_vd, sp_safe = straight_path(vis_entry, vis_target, vessel_tree)
    
    state = env.reset(vis_entry, vis_target)
    done = False
    while not done:
        action = agent.act(state, False)
        state, _, done, _ = env.step(action)
    cp_pts = np.array(env.path)
    cp_len = np.sum(np.linalg.norm(np.diff(cp_pts, axis=0), axis=1))
    cp_vd = min(vessel_tree.query(pt)[0] for pt in cp_pts)
    
    print(f"    Direct: {direct_dist:.1f}mm")
    print(f"    Straight: {sp_vd:.1f}mm clearance - {'SAFE' if sp_safe else 'UNSAFE'}")
    print(f"    Q-Learning: {cp_vd:.1f}mm clearance, {cp_len:.1f}mm path")
    
    # =========================================================================
    # PYBULLET VISUALIZATION
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
    
    # =========================================================================
    # FRANKA PANDA WITH IK
    # =========================================================================
    print("    Loading Franka Panda...")
    
    panda = p.loadURDF("franka_panda/panda.urdf", ROBOT_POS, useFixedBase=True)
    
    # End effector link index
    EE_LINK = 11
    NUM_JOINTS = 7
    
    # Get joint limits
    joint_limits_low = []
    joint_limits_high = []
    for i in range(NUM_JOINTS):
        info = p.getJointInfo(panda, i)
        joint_limits_low.append(info[8])
        joint_limits_high.append(info[9])
    
    # Starting position: above the entry point
    start_pos = entry_world + np.array([0, 0, 0.15])  # 15cm above entry
    
    # Desired orientation: pointing DOWN (z-axis pointing down)
    # Rotation: 180 degrees around X axis
    desired_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    def solve_ik(target_pos, target_orn):
        """Solve IK for target position and orientation"""
        joint_poses = p.calculateInverseKinematics(
            panda,
            EE_LINK,
            target_pos.tolist(),
            target_orn,
            lowerLimits=joint_limits_low,
            upperLimits=joint_limits_high,
            jointRanges=[7]*NUM_JOINTS,
            restPoses=[0, -0.5, 0, -2, 0, 2, 0.785],
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        return joint_poses[:NUM_JOINTS]
    
    def set_joints(joint_poses):
        """Set robot joint positions"""
        for i in range(NUM_JOINTS):
            p.resetJointState(panda, i, joint_poses[i])
        p.resetJointState(panda, 9, 0.02)
        p.resetJointState(panda, 10, 0.02)
    
    # Move to start position
    print("    Moving to start position...")
    start_joints = solve_ik(start_pos, desired_orn)
    set_joints(start_joints)
    
    # Needle attached to gripper
    NEEDLE_LEN = 0.10
    
    needle_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.003, length=NEEDLE_LEN,
                                      rgbaColor=[0.8, 0.8, 0.85, 1])
    needle_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.003, height=NEEDLE_LEN)
    needle_body = p.createMultiBody(0.01, needle_col, needle_vis, [0, 0, 0])
    
    # Constraint to attach needle to gripper
    needle_constraint = p.createConstraint(
        panda, EE_LINK,
        needle_body, -1,
        p.JOINT_FIXED,
        jointAxis=[0, 0, 1],
        parentFramePosition=[0, 0, 0.05],  # Offset from gripper
        childFramePosition=[0, 0, NEEDLE_LEN/2]
    )
    
    # Tip marker
    tip_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[1, 0.2, 0.2, 1])
    tip_body = p.createMultiBody(0, -1, tip_vis, [0, 0, 0])
    
    def update_tip():
        """Update needle tip marker position"""
        needle_state = p.getBasePositionAndOrientation(needle_body)
        pos = np.array(needle_state[0])
        orn = needle_state[1]
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        tip_pos = pos - rot @ np.array([0, 0, NEEDLE_LEN/2])
        p.resetBasePositionAndOrientation(tip_body, tip_pos.tolist(), [0, 0, 0, 1])
        return tip_pos
    
    # Settle simulation
    for _ in range(50):
        p.stepSimulation()
        update_tip()
    
    # Check position
    ee_state = p.getLinkState(panda, EE_LINK)
    ee_pos = np.array(ee_state[0])
    tip_pos = update_tip()
    
    print(f"    End effector: {ee_pos}")
    print(f"    Needle tip: {tip_pos}")
    print(f"    Entry point: {entry_world}")
    
    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=35,
        cameraPitch=-25,
        cameraTargetPosition=[0.3, 0.25, 0.4]
    )
    
    print("\n" + "="*70)
    print("READY - Robot arm will move during insertion!")
    print("="*70)
    
    input("\n>>> Press ENTER to start surgery...")
    
    # =========================================================================
    # PHASE 1: APPROACH - Move from above to HOVER above entry point
    # Needle tip stays ABOVE the skull, not touching yet
    # =========================================================================
    print("\n  Phase 1: Approaching entry point...")
    print("    Watch the robot arm move to position...\n")
    
    # Hover position: needle tip 2cm ABOVE the entry point
    hover_height = 0.02  # 2cm above entry
    hover_pos = entry_world + np.array([0, 0, hover_height])
    
    approach_steps = 50
    for i in range(approach_steps + 1):
        t = i / approach_steps
        # Move from start to HOVER position (not entry!)
        current_tip_pos = start_pos + t * (hover_pos - start_pos)
        
        # EE is above the tip
        ee_target = current_tip_pos + np.array([0, 0, NEEDLE_LEN + 0.03])
        
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        
        for _ in range(5):
            p.stepSimulation()
        update_tip()
        
        time.sleep(0.05)
        pct = (i / approach_steps) * 100
        print(f"\r    Approaching entry point: {pct:.0f}%", end="")
    
    print("\n\n    ✓ Needle is hovering above entry point!")
    print("    >>> Look at the simulation - needle tip is ABOVE the skull")
    input("\n    >>> Press ENTER to begin insertion into the brain...")
    
    # =========================================================================
    # PHASE 2: INSERTION - First penetrate entry, then follow Q-learning path
    # =========================================================================
    print("\n  Phase 2: Inserting needle along Q-learning path...")
    print("    Monitoring vessel clearance in real-time...\n")
    
    # First: Move from hover to entry point (penetrating skull)
    print("    Penetrating skull at entry point...")
    hover_pos = entry_world + np.array([0, 0, 0.02])
    
    penetrate_steps = 15
    for i in range(penetrate_steps + 1):
        t = i / penetrate_steps
        current_tip = hover_pos + t * (entry_world - hover_pos)
        ee_target = current_tip + np.array([0, 0, NEEDLE_LEN + 0.03])
        
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        
        for _ in range(3):
            p.stepSimulation()
        update_tip()
        time.sleep(0.08)
    
    print("    ✓ Entry point reached - now following Q-learning path\n")
    
    # Now follow the Q-learning path
    min_clearance_during_insertion = float('inf')
    collision_detected = False
    
    for i, brain_pt in enumerate(cp_pts):
        tip_target = to_world(brain_pt)
        
        # EE position: offset from tip by needle length
        ee_target = tip_target + np.array([0, 0, NEEDLE_LEN + 0.03])
        
        # Solve IK and move
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        
        for _ in range(3):
            p.stepSimulation()
        actual_tip = update_tip()
        
        # Check vessel clearance at current position
        clearance = vessel_tree.query(brain_pt)[0]
        min_clearance_during_insertion = min(min_clearance_during_insertion, clearance)
        
        # Determine status
        if clearance < 4.0:
            status = "⚠️  WARNING - TOO CLOSE!"
            collision_detected = True
            status_color = "\033[91m"  # Red
        elif clearance < 6.0:
            status = "⚡ CAUTION"
            status_color = "\033[93m"  # Yellow
        else:
            status = "✓ SAFE"
            status_color = "\033[92m"  # Green
        
        reset_color = "\033[0m"
        
        # SLOW animation - 0.15 seconds per step
        time.sleep(0.15)
        
        depth_mm = np.linalg.norm(brain_pt - vis_entry)
        pct = (i + 1) / len(cp_pts) * 100
        print(f"\r    Depth: {depth_mm:5.1f}mm | Clearance: {clearance:4.1f}mm | {status_color}{status}{reset_color} | Progress: {pct:5.1f}%   ", end="")
    
    # =========================================================================
    # PAUSE AT TUMOR - Let user see the needle at target
    # =========================================================================
    print("\n\n  " + "="*50)
    print("  ✓ TUMOR REACHED!")
    print("  " + "="*50)
    print(f"    Minimum clearance during insertion: {min_clearance_during_insertion:.1f}mm")
    if not collision_detected:
        print("    ✓ No vessel collisions detected!")
    print("\n    >>> Needle is now at the tumor. Look at the simulation!")
    print("    >>> The needle tip (red) should be at the green TUMOR marker.")
    
    input("\n    >>> Press ENTER when ready to retract the needle...")
    
    # =========================================================================
    # PHASE 3: RETRACT - Same speed as insertion with clearance monitoring
    # =========================================================================
    print("\n  Phase 3: Retracting needle slowly...")
    
    # Retract along the same path in reverse
    retract_pts = cp_pts[::-1]  # Reverse the path
    
    for i, brain_pt in enumerate(retract_pts):
        tip_target = to_world(brain_pt)
        ee_target = tip_target + np.array([0, 0, NEEDLE_LEN + 0.03])
        
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        
        for _ in range(3):
            p.stepSimulation()
        update_tip()
        
        # Check clearance during retraction
        clearance = vessel_tree.query(brain_pt)[0]
        
        if clearance < 4.0:
            status = "⚠️  WARNING"
        elif clearance < 6.0:
            status = "⚡ CAUTION"
        else:
            status = "✓ SAFE"
        
        # SLOW retraction - 0.12 seconds per step
        time.sleep(0.12)
        
        depth_mm = np.linalg.norm(brain_pt - vis_entry)
        pct = (i + 1) / len(retract_pts) * 100
        print(f"\r    Depth: {depth_mm:5.1f}mm | Clearance: {clearance:4.1f}mm | {status} | Progress: {pct:5.1f}%   ", end="")
    
    # Exit skull - move from entry to hover
    print("\n\n    Exiting skull...")
    hover_pos = entry_world + np.array([0, 0, 0.02])
    
    for i in range(15):
        t = i / 15
        current_tip = entry_world + t * (hover_pos - entry_world)
        ee_target = current_tip + np.array([0, 0, NEEDLE_LEN + 0.03])
        
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        
        for _ in range(3):
            p.stepSimulation()
        update_tip()
        time.sleep(0.06)
    
    # Final move back to start position
    print("\n    Moving to safe position...")
    for i in range(30):
        t = i / 30
        current_pos = entry_world + t * (start_pos - entry_world)
        ee_target = current_pos + np.array([0, 0, NEEDLE_LEN + 0.05])
        
        joint_poses = solve_ik(ee_target, desired_orn)
        set_joints(joint_poses)
        
        for _ in range(3):
            p.stepSimulation()
        update_tip()
        
        time.sleep(0.03)
    
    print("    Surgery complete - needle safely retracted!")
    
    # Results
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                  SURGERY COMPLETE                             ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Training:       {train_sr:5.1f}%                                    ║
    ║  Generalization: {gen_rate:5.1f}% ({gen_ok}/{len(test_targets)})                                ║
    ║  Q-states:       {len(agent.q):5d}                                      ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Straight: {sp_vd:5.1f}mm clearance {'✓' if sp_safe else '✗'}                           ║
    ║  Q-Learn:  {cp_vd:5.1f}mm clearance, {cp_len:5.1f}mm path                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    input(">>> Press ENTER to exit...")
    p.disconnect()


if __name__ == "__main__":
    main()
