import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import os

class QuadrupedPyBulletEnv(gym.Env):
    """
    A PyBullet Gym environment for a quadruped robot (e.g., Mini Cheetah).
    This environment handles the physics simulation, state observation,
    action application, and reward calculation.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}
    
    def __init__(self, render_mode='human', urdf_path="quadruped_model.urdf"):
        super().__init__()
        self.render_mode = render_mode
        self.urdf_path = urdf_path

        # Connect to the PyBullet physics server
        if self.render_mode == 'human':
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        # Set up a few physics parameters
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(fixedTimeStep=1./240., numSolverIterations=100, physicsClientId=self.client_id)

        # Action space: 12 joint position targets, normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation space: 12 joint positions, 12 joint velocities, 3 base orientation (r,p,y), 3 base angular vel, 3 base lin vel.
        # Total: 12 + 12 + 3 + 3 + 3 = 33
        observation_dim = 33
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
        
        self.last_base_pos = None
        self.joint_ids = []
        self.joint_limits_low = []
        self.joint_limits_high = []
        
        self.reset()
        
    def _load_robot_and_plane(self):
        # Load a ground plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # Load the robot URDF
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {self.urdf_path}")
            
        self.robot = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0.3], useFixedBase=False, physicsClientId=self.client_id)
        
        # Identify the controllable joints
        self.num_joints = p.getNumJoints(self.robot, physicsClientId=self.client_id)
        self.joint_ids = []
        self.joint_limits_low = []
        self.joint_limits_high = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client_id)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.joint_ids.append(i)
                self.joint_limits_low.append(info[8])
                self.joint_limits_high.append(info[9])

        self.joint_limits_low = np.array(self.joint_limits_low)
        self.joint_limits_high = np.array(self.joint_limits_high)

    def _get_obs(self):
        # Get base state
        pos, ori = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(ori)

        # Get joint states
        joint_positions = []
        joint_velocities = []
        for joint_id in self.joint_ids:
            state = p.getJointState(self.robot, joint_id, physicsClientId=self.client_id)
            joint_positions.append(state[0])
            joint_velocities.append(state[1])
            
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            [roll, pitch, yaw],
            ang_vel,
            lin_vel
        ]).astype(np.float32)
        
        return observation

    def _apply_action(self, action):
        # Scale DDPG's [-1, 1] action to actual joint limits
        scaled_action = self.joint_limits_low + 0.5 * (action + 1.0) * \
                        (self.joint_limits_high - self.joint_limits_low)
        
        # Apply position control to joints
        p.setJointMotorControlArray(
            self.robot,
            self.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=scaled_action,
            forces=[500.0] * len(self.joint_ids),
            physicsClientId=self.client_id
        )
        
    def reset(self, seed=None, options=None):
        # The fix for the TypeError is here: removing the call to super().reset(seed=seed)
        
        # Reset physics and load plane/robot
        p.resetSimulation(physicsClientId=self.client_id)
        self._load_robot_and_plane()
        
        # Initial joint positions (example for a standing pose)
        initial_joint_positions = [0.0, 0.7, -1.4,  # Front Left
                                   0.0, 0.7, -1.4,  # Front Right
                                   0.0, 0.7, -1.4,  # Rear Left
                                   0.0, 0.7, -1.4]  # Rear Right
        
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot, joint_id, initial_joint_positions[i], physicsClientId=self.client_id)
        
        # Allow the robot to settle on the ground
        for _ in range(240): # 1 second of simulation time
            p.stepSimulation(physicsClientId=self.client_id)
        
        self.last_base_pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self._apply_action(action)
        
        # Simulate physics for a few steps per action to allow dynamics to play out
        for _ in range(24): # 240Hz physics, so 24 steps = 0.1 seconds per action
            p.stepSimulation(physicsClientId=self.client_id)

        # Get new observation
        observation = self._get_obs()

        # Calculate reward
        current_base_pos, current_base_ori = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(current_base_ori)
        
        forward_reward = (current_base_pos[0] - self.last_base_pos[0]) * 10.0
        # Penalize large body tilt
        orientation_penalty = -10.0 * (abs(roll) + abs(pitch))
        
        reward = forward_reward + orientation_penalty
        
        self.last_base_pos = current_base_pos

        # Check for done condition (fallen over)
        done = False
        if current_base_pos[2] < 0.15 or abs(roll) > np.pi/2 or abs(pitch) > np.pi/2:
            done = True
            reward -= 100.0 # Large penalty for falling

        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            # Implement camera rendering here if needed
            pass
        # 'human' mode is handled by p.connect(p.GUI)

    def close(self):
        p.disconnect(self.client_id)