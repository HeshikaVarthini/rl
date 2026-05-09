'''import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 3.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.obstacle_id = None
        self.step_counter = 0
        self.sensor_lines = []
        super().__init__(render=render)
        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)

    def reset(self):
        self.step_counter = 0
        obs = super().reset()
        self._add_obstacle()
        return self._get_observation_with_sensors(obs)

    def _add_obstacle(self):
        if self.obstacle_id is not None:
            p.removeBody(self.obstacle_id)
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[1], base_matrix[2]])
        obstacle_position = np.array(base_position) + forward_vector * 2.5
        obstacle_position[2] = 0.25
        obstacle_urdf = pybullet_data.getDataPath() + "/cube.urdf"
        self.obstacle_id = p.loadURDF(
            obstacle_urdf,
            obstacle_position.tolist(),
            useFixedBase=True,
            globalScaling=0.5
        )
        p.changeVisualShape(self.obstacle_id, -1, rgbaColor=[1, 0, 0, 1])

    def _get_observation_with_sensors(self, base_obs):
        sensor_data = []
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[1], base_matrix[2]])
        right_vector = np.array([base_matrix[3], base_matrix[4], base_matrix[5]])
        up_vector = np.array([base_matrix[6], base_matrix[7], base_matrix[8]])
        for line_id in self.sensor_lines:
            p.removeUserDebugItem(line_id)
        self.sensor_lines = []
        sensor_height = 0.3
        start = np.array(base_position) + sensor_height * up_vector
        for angle in self.sensor_angles:
            ray_dir = (
                np.cos(angle) * forward_vector +
                np.sin(angle) * right_vector
            )
            ray_dir /= np.linalg.norm(ray_dir)
            end = start + ray_dir * self.sensor_range
            result = p.rayTest(start, end)[0]
            hit_object_id = result[0]
            if hit_object_id == self.obstacle_id:
                hit_fraction = result[2]
            else:
                hit_fraction = 1.0
            sensor_data.append(hit_fraction)
            if hasattr(self, '_is_render') and self._is_render:
                color = [1, 0, 0] if hit_object_id == self.obstacle_id else [0, 1, 0]
                shortened_end = start + ray_dir * self.sensor_range * hit_fraction
                line_id = p.addUserDebugLine(
                    start, shortened_end, color,
                    lifeTime=0.1,
                    lineWidth=3
                )
                self.sensor_lines.append(line_id)
        return np.concatenate([base_obs, sensor_data])

    def step(self, action):
        self.step_counter += 1
        base_obs, reward, done, info = super().step(action)
        contact_points = p.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True
        return self._get_observation_with_sensors(base_obs), reward, done, info'''
'''import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 3.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.obstacle_id = None
        self.step_counter = 0
        self.sensor_lines = []
        super().__init__(render=render)
        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)
        self.prev_base_pos = None

    def reset(self):
        self.step_counter = 0
        obs = super().reset()
        self._add_obstacle()
        self.prev_base_pos = p.getBasePositionAndOrientation(self.minitaur.quadruped)[0]
        return self._get_observation_with_sensors(obs)

    def _add_obstacle(self):
        if self.obstacle_id is not None:
            p.removeBody(self.obstacle_id)
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[1], base_matrix[2]])
        obstacle_position = np.array(base_position) + forward_vector * 2.5
        obstacle_position[2] = 0.25
        obstacle_urdf = pybullet_data.getDataPath() + "/cube.urdf"
        self.obstacle_id = p.loadURDF(
            obstacle_urdf,
            obstacle_position.tolist(),
            useFixedBase=True,
            globalScaling=0.5
        )
        p.changeVisualShape(self.obstacle_id, -1, rgbaColor=[1, 0, 0, 1])

    def _get_observation_with_sensors(self, base_obs):
        sensor_data = []
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[1], base_matrix[2]])
        right_vector = np.array([base_matrix[3], base_matrix[4], base_matrix[5]])
        up_vector = np.array([base_matrix[6], base_matrix[7], base_matrix[8]])
        for line_id in self.sensor_lines:
            p.removeUserDebugItem(line_id)
        self.sensor_lines = []
        sensor_height = 0.3
        start = np.array(base_position) + sensor_height * up_vector
        for angle in self.sensor_angles:
            ray_dir = (
                np.cos(angle) * forward_vector +
                np.sin(angle) * right_vector
            )
            ray_dir /= np.linalg.norm(ray_dir)
            end = start + ray_dir * self.sensor_range
            result = p.rayTest(start, end)[0]
            hit_object_id = result[0]
            if hit_object_id == self.obstacle_id:
                hit_fraction = result[2]
            else:
                hit_fraction = 1.0
            sensor_data.append(hit_fraction)
        return np.concatenate([base_obs, sensor_data])

    def step(self, action):
        self.step_counter += 1
        base_obs, reward, done, info = super().step(action)
        current_base_pos = p.getBasePositionAndOrientation(self.minitaur.quadruped)[0]
        if self.prev_base_pos is None:
            self.prev_base_pos = current_base_pos
        # Forward progress (X axis)
        forward_progress = current_base_pos[0] - self.prev_base_pos[0]
        self.prev_base_pos = current_base_pos
        # Get sensor readings
        sensors = self._get_observation_with_sensors(base_obs)[-8:]
        close_penalty = -2.0 if sensors[4] < 0.6 else 0.0
        contact_points = p.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True
        # Positive reward for moving forward, mild penalty for getting too close to the obstacle
        reward += 3.0 * forward_progress + close_penalty
        return self._get_observation_with_sensors(base_obs), reward, done, info
'''
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

class MinitaurEnv(gym.Env):
    """
    Wraps your walking controller with RL for obstacle avoidance.
    RL chooses high-level actions: 0 = forward, 1 = turn left, 2 = turn right.
    """
    def __init__(self, render=False):
        super(MinitaurEnv, self).__init__()
        self.render_mode = render
        if self.render_mode:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0/240.0, physicsClientId=self.client)

        # Plane + robot
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("quadruped/minitaur.urdf", [0, 0, 0.2])

        # Add obstacle
        self.obstacle = p.loadURDF("cube_small.urdf", [2, 0, 0.1], globalScaling=2)

        # CORRECTED: Observation space should be 13 (robot state) + 5 (sensors) = 18
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13 + 5,), dtype=np.float32
        )
        # Action: discrete {forward, left, right}
        self.action_space = spaces.Discrete(3)

        self.max_steps = 500
        self.step_counter = 0

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0/240.0, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("quadruped/minitaur.urdf", [0, 0, 0.2])
        self.obstacle = p.loadURDF("cube_small.urdf", [2, 0, 0.1], globalScaling=2)
        self.step_counter = 0
        
        # Let the robot settle
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_obs()

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        
        # The robot state has 13 components (not 12):
        # 3 position + 4 orientation + 3 linear velocity + 3 angular velocity = 13
        state = np.array(pos + orn + lin_vel + ang_vel, dtype=np.float32)

        # Add 5 ray sensors for obstacle distances (in robot frame)
        ray_angles = [-0.5, -0.25, 0, 0.25, 0.5]
        ray_results = []
        
        # Get robot orientation as quaternion and convert to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        for angle in ray_angles:
            from_pos = np.array(pos) + [0, 0, 0.1]
            # Calculate direction in robot frame then transform to world frame
            ray_dir_local = [np.cos(angle), np.sin(angle), 0]
            ray_dir_world = np.dot(rot_matrix, ray_dir_local)
            to_pos = from_pos + ray_dir_world * 2.0
            res = p.rayTest(from_pos, to_pos)[0]
            # Use hit fraction as distance measurement
            ray_results.append(res[2] if res[2] != -1 else 1.0)
            
        # Combine state (13) and ray results (5) = 18 total
        return np.concatenate([state, np.array(ray_results, dtype=np.float32)])

    def _apply_gait(self, action):
        """
        Your existing gait controller goes here.
        RL chooses high-level action, gait executes it.
        """
        # Improved gait patterns with smoother transitions
        if action == 0:   # forward
            target_angles = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        elif action == 1: # turn left
            target_angles = [0.4, -0.4, -0.3, 0.3, 0.4, -0.4, -0.3, 0.3]
        else:             # turn right
            target_angles = [-0.3, 0.3, 0.4, -0.4, -0.3, 0.3, 0.4, -0.4]

        for j in range(8):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=target_angles[j], force=20)

    def step(self, action):
        # Execute multiple simulation steps for stability
        for _ in range(4):
            self._apply_gait(action)
            p.stepSimulation()
            
        obs = self._get_obs()

        # Reward: move forward in x, penalize collisions
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        reward = pos[0] * 10.0  # encourage forward movement

        # Penalty for falling
        if pos[2] < 0.1:  # robot is too low (fell)
            reward -= 100
            done = True
        else:
            # Collision penalty
            contact = len(p.getContactPoints(self.robot, self.obstacle)) > 0
            if contact:
                reward -= 50
                done = True
            else:
                done = False

        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            done = True
            
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect(self.client)


# ------------------ Training ------------------
if __name__ == "__main__":
    env = DummyVecEnv([lambda: MinitaurEnv(render=False)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("ppo_minitaur_obstacle")

    # Evaluate
    test_env = MinitaurEnv(render=True)
    obs = test_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = test_env.step(action)
        time.sleep(1. / 60.)
        if done:
            obs = test_env.reset()