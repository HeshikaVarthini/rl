import gym
import numpy as np
import pybullet_data
from minitaur_env import MinitaurBulletEnv


class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        super().__init__(render=render)
        self.obstacle_id = None
        self._add_obstacle()

    def _add_obstacle(self):
        if self.obstacle_id is not None:
            p.removeBody(self.obstacle_id)

        base_position, _ = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        dist = np.random.uniform(1.5, 2.5)
        offset = np.array([np.cos(angle), np.sin(angle), 0]) * dist
        obstacle_position = np.array(base_position) + offset
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
        sensor_obs = np.zeros(4)
        base_position, _ = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        directions = [
            np.array([1, 0, 0]),   # front
            np.array([-1, 0, 0]),  # back
            np.array([0, 1, 0]),   # left
            np.array([0, -1, 0])   # right
        ]
        for i, dir_vec in enumerate(directions):
            from_pos = base_position + np.array([0, 0, 0.2])
            to_pos = from_pos + dir_vec * 2.0
            result = p.rayTest(from_pos, to_pos)[0]
            hit_fraction = result[2]
            sensor_obs[i] = 1.0 - hit_fraction if result[0] != -1 else 0.0
        return np.concatenate([base_obs, sensor_obs])

    def reset(self):
        self.step_counter = 0
        obs = super().reset()
        self._add_obstacle()
        self.prev_pos = p.getBasePositionAndOrientation(self.minitaur.quadruped)[0]
        return self._get_observation_with_sensors(obs)

    def step(self, action):
        self.step_counter += 1
        base_obs, reward, done, info = super().step(action)

        contact_points = p.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True
            info["collision"] = True
        else:
            info["collision"] = False

        current_pos = p.getBasePositionAndOrientation(self.minitaur.quadruped)[0]
        if hasattr(self, "prev_pos"):
            delta_x = current_pos[0] - self.prev_pos[0]
            reward += 5.0 * delta_x
        self.prev_pos = current_pos

        return self._get_observation_with_sensors(base_obs), reward, done, info
