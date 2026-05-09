import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import pybullet_envs  # loads Laikago, Minitaur, etc.
import time

class QuadrupedWalkEnv(gym.Env):
    def __init__(self, render=False):
        super(QuadrupedWalkEnv, self).__init__()
        self.render = render
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, 0.5])

        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, 0.5])
        self.step_counter = 0
        return self._get_obs()

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))]
        return np.array(pos + orn + lin_vel + ang_vel + joint_states[:4])  # You can expand this

    def step(self, action):
        max_force = 20
        for i, act in enumerate(action):
            if i < p.getNumJoints(self.robot):
                p.setJointMotorControl2(
                    bodyIndex=self.robot,
                    jointIndex=i,
                    controlMode=p.TORQUE_CONTROL,
                    force=act * max_force,
                )
        p.stepSimulation()
        time.sleep(1./240.) if self.render else None
        self.step_counter += 1

        obs = self._get_obs()
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        reward = pos[0]  # reward for forward motion
        done = self.step_counter > 1000 or pos[2] < 0.2  # fall down or timeout
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)
