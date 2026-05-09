import gym
import numpy as np
import pybullet as p
import pybullet_data
import os

class QuadrupedEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        self.time_step = 1. / 240.
        self.max_force = 20
        self.control_steps_per_action = 4  # Number of sim steps per action

        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.robot = None

        # Action space: 12 motors (normalized [-1,1])
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        
        # Observation space: 37 elements (see _get_obs())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")
        
        start_pos = [0, 0, 0.48]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF("laikago/laikago_toes.urdf", start_pos, start_ori)

        self.motor_ids = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        
        # Reset joints to neutral position
        for i, motor_id in enumerate(self.motor_ids):
            p.resetJointState(self.robot, motor_id, targetValue=0)
            
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1)
        # Scale actions to reasonable joint angles (in radians)
        target_positions = action * 1.0  # Now ±1.0 radian (~57 degrees)
        
        for i, motor_id in enumerate(self.motor_ids):
            p.setJointMotorControl2(
                self.robot, motor_id, p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                force=self.max_force,
                positionGain=0.5,
                velocityGain=0.1
            )

        for _ in range(self.control_steps_per_action):
            p.stepSimulation()

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_done(obs)
        info = {}
        
        return obs, reward, done, info

    def _get_obs(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        joint_states = p.getJointStates(self.robot, self.motor_ids)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        
        # Convert quaternion to Euler angles (3 values instead of 4)
        euler = p.getEulerFromQuaternion(ori)
        
        obs = np.array(
            list(pos) +          # 3
            list(euler) +        # 3
            list(lin_vel) +      # 3
            list(ang_vel) +     # 3
            joint_pos +          # 12
            joint_vel,          # 12
            dtype=np.float32    # Total: 3+3+3+3+12+12 = 36
        )
        return obs

    def _compute_reward(self, obs, action):
        forward_vel = obs[3]  # x-direction linear velocity (now at index 3 after pos+euler)
        
        # Penalize large joint velocities and actions
        joint_vel_penalty = 0.01 * np.sum(np.square(obs[24:36]))
        action_penalty = 0.01 * np.sum(np.square(action))
        
        # Penalize body rotation (want to stay upright)
        pitch_roll_penalty = 0.1 * (abs(obs[4]) + abs(obs[5]))  # pitch and roll
        
        reward = forward_vel - joint_vel_penalty - action_penalty - pitch_roll_penalty
        return reward

    def _check_done(self, obs):
        z_pos = obs[2]
        pitch, roll = obs[4], obs[5]
        
        # Episode ends if robot falls or tilts too much
        done = (z_pos < 0.2) or (abs(pitch) > 0.8) or (abs(roll) > 0.8)
        return done

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None