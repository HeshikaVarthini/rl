import os
import time
import numpy as np

import gym
from gym import spaces

import pybullet as p
import pybullet_data

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
except Exception as import_error:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from import_error


# Import your environment class
from minisenchum1 import MinitaurWithSensors


class ActionBiasScaleWrapper(gym.Wrapper):
    """
    Stabilize gait by scaling actions and adding knee flexion bias.
    Keeps final actions within the original Box limits.
    """
    def __init__(self, env: gym.Env, knee_bias: float = 0.3, scale: float = 0.8):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        self.knee_bias = float(knee_bias)
        self.scale = float(scale)
        self._last_action = np.zeros(env.action_space.shape, dtype=np.float32)

        # Precompute bias vector for knee joints
        num_joints = env.action_space.shape[0]
        bias = np.zeros(num_joints, dtype=np.float32)
        knee_indices = [1, 3, 5, 7] if num_joints >= 8 else list(range(1, num_joints, 2))
        for idx in knee_indices:
            bias[idx] = self.knee_bias
        self._bias = bias

    def reset(self, **kwargs):
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.tanh(action)  # ensure within [-1, 1]
        action = self.scale * action + self._bias
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self._last_action = action
        return self.env.step(action)


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds reward for forward progress and obstacle avoidance, plus posture and smoothness terms
    to encourage a stable trot with lifted knees.

    Observation is unchanged (base obs + 8 sensor fractions). Reward becomes:
        reward = forward_gain * delta_x
                 - proximity_penalty_weight * proximity_penalty
                 - collision_penalty
                 + alive_bonus
    where proximity_penalty uses front/left/right sensors.
    """

    def __init__(self, env: gym.Env,
                 forward_gain: float = 2.0,
                 proximity_weight: float = 5.0,
                 collision_cost: float = 100.0,
                 alive_bonus: float = 0.02,
                 obstacle_threshold_m: float = 0.8,
                 posture_weight: float = 1.0,
                 smooth_weight: float = 0.05):
        super().__init__(env)
        self.forward_gain = forward_gain
        self.proximity_weight = proximity_weight
        self.collision_cost = collision_cost
        self.alive_bonus = alive_bonus
        self.obstacle_threshold_m = obstacle_threshold_m
        self.posture_weight = posture_weight
        self.smooth_weight = smooth_weight

        self._last_base_x: float = 0.0
        self._last_base_pos = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Expect last 8 obs entries to be sensor fractions in [0,1]
        assert isinstance(self.observation_space, spaces.Box)
        assert self.observation_space.shape[0] >= 8, "Observation must include 8 sensor readings."

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_base_x = self._get_base_x()
        base_pos, _ = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        self._last_base_pos = np.array(base_pos, dtype=np.float32)
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)

        # Forward progress aligned with robot heading
        base_pos, base_ori = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        disp = np.array(base_pos, dtype=np.float32) - self._last_base_pos
        self._last_base_pos = np.array(base_pos, dtype=np.float32)

        base_mat = p.getMatrixFromQuaternion(base_ori)
        body_forward = np.array([base_mat[0], base_mat[3], base_mat[6]], dtype=np.float32)
        norm = np.linalg.norm(body_forward) + 1e-8
        body_forward /= norm
        forward_progress = float(np.dot(disp, body_forward))

        # Keep delta_x for debugging
        current_x = float(base_pos[0])
        delta_x = current_x - self._last_base_x
        self._last_base_x = current_x

        forward_reward = self.forward_gain * forward_progress

        # Proximity penalty using front/left/right sensors (dist fractions, convert to meters)
        sensor_fractions = obs[-8:]
        # Angles: [-pi, -3pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3pi/4]
        front_fraction = sensor_fractions[4]
        left_fraction = sensor_fractions[2]
        right_fraction = sensor_fractions[6]

        # Convert fraction to distance in meters using env.sensor_range
        sensor_range_m = getattr(self.env, 'sensor_range', 2.0)
        front_m = float(front_fraction) * sensor_range_m
        left_m = float(left_fraction) * sensor_range_m
        right_m = float(right_fraction) * sensor_range_m

        min_front_lr = min(front_m, left_m, right_m)
        proximity_shortfall = max(0.0, self.obstacle_threshold_m - min_front_lr)
        proximity_penalty = self.proximity_weight * proximity_shortfall

        # Collision penalty if robot touches obstacle
        collision_happened = self._has_collision_with_obstacle()
        collision_penalty = self.collision_cost if collision_happened else 0.0

        # Posture penalty (discourage roll/pitch)
        _, base_ori = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        roll, pitch, _ = p.getEulerFromQuaternion(base_ori)
        posture_penalty = self.posture_weight * (abs(roll) + abs(pitch))

        # Smoothness penalty (discourage large action deltas)
        if isinstance(action, np.ndarray):
            action_np = action.astype(np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)
        smooth_penalty = self.smooth_weight * float(np.linalg.norm(action_np - self._last_action, ord=1))
        self._last_action = action_np

        shaped_reward = (
            forward_reward
            - proximity_penalty
            - collision_penalty
            - posture_penalty
            - smooth_penalty
            + self.alive_bonus
        )

        # Optionally terminate on collision
        if collision_happened:
            done = True

        # Expose diagnostics
        info = dict(info)
        info.update({
            'delta_x': delta_x,
            'forward_progress': forward_progress,
            'forward_reward': forward_reward,
            'proximity_penalty': proximity_penalty,
            'collision_penalty': float(collision_penalty),
            'front_m': front_m,
            'left_m': left_m,
            'right_m': right_m,
            'posture_penalty': posture_penalty,
            'smooth_penalty': smooth_penalty,
        })

        return obs, shaped_reward, done, info

    def _get_base_x(self) -> float:
        # Compatible with underlying MinitaurWithSensors that uses pybullet directly
        base_pos, _ = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        return float(base_pos[0])

    def _has_collision_with_obstacle(self) -> bool:
        try:
            contact_points = p.getContactPoints(self.env.minitaur.quadruped, self.env.obstacle_id)
            return len(contact_points) > 0
        except Exception:
            return False


def make_env(render: bool = False) -> gym.Env:
    base_env = MinitaurWithSensors(render=render)
    # Apply action bias/scale first, then reward shaping
    gait_env = ActionBiasScaleWrapper(base_env, knee_bias=0.3, scale=0.8)
    wrapped_env = RewardShapingWrapper(gait_env)
    return wrapped_env


def main():
    num_envs = int(os.environ.get('NUM_ENVS', '1'))
    total_timesteps = int(os.environ.get('TOTAL_STEPS', str(400_000)))
    save_dir = os.environ.get('SAVE_DIR', os.path.join('checkpoints', 'ppo_minitaur_obstacle'))
    os.makedirs(save_dir, exist_ok=True)

    if num_envs > 1:
        def _make():
            return make_env(render=False)
        vec_env = SubprocVecEnv([_make for _ in range(num_envs)])
    else:
        vec_env = DummyVecEnv([lambda: make_env(render=False)])

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000 // num_envs,
        save_path=save_dir,
        name_prefix='ppo_minitaur_obs'
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // max(1, num_envs),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=os.path.join(save_dir, 'tb')
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    final_path = os.path.join(save_dir, 'ppo_minitaur_obs_final')
    model.save(final_path)
    print(f"Saved final model to: {final_path}")


if __name__ == '__main__':
    main()
'''
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import gym
from gym import spaces

import pybullet as p
import pybullet_data

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
except Exception as import_error:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from import_error


# Import your environment class
from minisenchum1 import MinitaurWithSensors


class ActionBiasScaleWrapper(gym.Wrapper):
    """
    Stabilize gait by scaling actions and adding knee flexion bias.
    Keeps final actions within the original Box limits.
    """
    def __init__(self, env: gym.Env, knee_bias: float = 0.3, scale: float = 0.8):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        self.knee_bias = float(knee_bias)
        self.scale = float(scale)
        self._last_action = np.zeros(env.action_space.shape, dtype=np.float32)

        # Precompute bias vector for knee joints
        num_joints = env.action_space.shape[0]
        bias = np.zeros(num_joints, dtype=np.float32)
        knee_indices = [1, 3, 5, 7] if num_joints >= 8 else list(range(1, num_joints, 2))
        for idx in knee_indices:
            bias[idx] = self.knee_bias
        self._bias = bias

    def reset(self, **kwargs):
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.tanh(action)  # ensure within [-1, 1]
        action = self.scale * action + self._bias
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self._last_action = action
        return self.env.step(action)


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds reward for forward progress and obstacle avoidance, plus posture and smoothness terms
    to encourage a stable trot with lifted knees.

    Observation is unchanged (base obs + 8 sensor fractions). Reward becomes:
        reward = forward_gain * delta_x
                 + obstacle_avoidance_bonus
                 + stability_bonus
                 - collision_penalty
                 + alive_bonus
    where obstacle_avoidance_bonus rewards staying away from obstacles.
    """

    def __init__(self, env: gym.Env,
                 forward_gain: float = 3.0,
                 obstacle_avoidance_bonus: float = 2.0,
                 collision_cost: float = 50.0,
                 alive_bonus: float = 0.1,
                 obstacle_threshold_m: float = 1.0,
                 stability_bonus: float = 0.5,
                 smooth_weight: float = 0.02):
        super().__init__(env)
        self.forward_gain = forward_gain
        self.obstacle_avoidance_bonus = obstacle_avoidance_bonus
        self.collision_cost = collision_cost
        self.alive_bonus = alive_bonus
        self.obstacle_threshold_m = obstacle_threshold_m
        self.stability_bonus = stability_bonus
        self.smooth_weight = smooth_weight

        self._last_base_x: float = 0.0
        self._last_base_pos = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Expect last 8 obs entries to be sensor fractions in [0,1]
        assert isinstance(self.observation_space, spaces.Box)
        assert self.observation_space.shape[0] >= 8, "Observation must include 8 sensor readings."

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_base_x = self._get_base_x()
        base_pos, _ = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        self._last_base_pos = np.array(base_pos, dtype=np.float32)
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)

        # Forward progress aligned with robot heading
        base_pos, base_ori = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        disp = np.array(base_pos, dtype=np.float32) - self._last_base_pos
        self._last_base_pos = np.array(base_pos, dtype=np.float32)

        base_mat = p.getMatrixFromQuaternion(base_ori)
        body_forward = np.array([base_mat[0], base_mat[3], base_mat[6]], dtype=np.float32)
        norm = np.linalg.norm(body_forward) + 1e-8
        body_forward /= norm
        forward_progress = float(np.dot(disp, body_forward))

        # Keep delta_x for debugging
        current_x = float(base_pos[0])
        delta_x = current_x - self._last_base_x
        self._last_base_x = current_x

        forward_reward = self.forward_gain * forward_progress

        # Obstacle avoidance bonus using front/left/right sensors (dist fractions, convert to meters)
        sensor_fractions = obs[-8:]
        # Angles: [-pi, -3pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3pi/4]
        front_fraction = sensor_fractions[4]
        left_fraction = sensor_fractions[2]
        right_fraction = sensor_fractions[6]

        # Convert fraction to distance in meters using env.sensor_range
        sensor_range_m = getattr(self.env, 'sensor_range', 2.0)
        front_m = float(front_fraction) * sensor_range_m
        left_m = float(left_fraction) * sensor_range_m
        right_m = float(right_fraction) * sensor_range_m

        min_front_lr = min(front_m, left_m, right_m)
        # Reward for staying away from obstacles (positive reward)
        obstacle_avoidance_reward = self.obstacle_avoidance_bonus * min(1.0, min_front_lr / self.obstacle_threshold_m)

        # Collision penalty if robot touches obstacle
        collision_happened = self._has_collision_with_obstacle()
        collision_penalty = self.collision_cost if collision_happened else 0.0

        # Stability bonus (reward for good posture - low roll/pitch)
        _, base_ori = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        roll, pitch, _ = p.getEulerFromQuaternion(base_ori)
        posture_angle = abs(roll) + abs(pitch)
        stability_reward = self.stability_bonus * max(0.0, 1.0 - posture_angle / (np.pi/4))  # Max reward when posture is good

        # Smoothness penalty (discourage large action deltas) - keep this as penalty
        if isinstance(action, np.ndarray):
            action_np = action.astype(np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)
        smooth_penalty = self.smooth_weight * float(np.linalg.norm(action_np - self._last_action, ord=1))
        self._last_action = action_np

        shaped_reward = (
            forward_reward
            + obstacle_avoidance_reward
            + stability_reward
            - collision_penalty
            - smooth_penalty
            + self.alive_bonus
        )

        # Optionally terminate on collision
        if collision_happened:
            done = True

        # Expose diagnostics
        info = dict(info)
        info.update({
            'delta_x': delta_x,
            'forward_progress': forward_progress,
            'forward_reward': forward_reward,
            'obstacle_avoidance_reward': obstacle_avoidance_reward,
            'stability_reward': stability_reward,
            'collision_penalty': float(collision_penalty),
            'front_m': front_m,
            'left_m': left_m,
            'right_m': right_m,
            'posture_angle': posture_angle,
            'smooth_penalty': smooth_penalty,
        })

        return obs, shaped_reward, done, info

    def _get_base_x(self) -> float:
        # Compatible with underlying MinitaurWithSensors that uses pybullet directly
        base_pos, _ = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        return float(base_pos[0])

    def _has_collision_with_obstacle(self) -> bool:
        try:
            contact_points = p.getContactPoints(self.env.minitaur.quadruped, self.env.obstacle_id)
            return len(contact_points) > 0
        except Exception:
            return False


def make_env(render: bool = False) -> gym.Env:
    base_env = MinitaurWithSensors(render=render)
    # Apply action bias/scale first, then reward shaping
    gait_env = ActionBiasScaleWrapper(base_env, knee_bias=0.3, scale=0.8)
    wrapped_env = RewardShapingWrapper(gait_env)
    return wrapped_env


def main():
    num_envs = int(os.environ.get('NUM_ENVS', '1'))
    total_timesteps = int(os.environ.get('TOTAL_STEPS', str(400_000)))
    save_dir = os.environ.get('SAVE_DIR', os.path.join('checkpoints', 'ppo_minitaur_obstacle'))
    os.makedirs(save_dir, exist_ok=True)

    if num_envs > 1:
        def _make():
            return make_env(render=False)
        vec_env = SubprocVecEnv([_make for _ in range(num_envs)])
    else:
        vec_env = DummyVecEnv([lambda: make_env(render=False)])

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000 // num_envs,
        save_path=save_dir,
        name_prefix='ppo_minitaur_obs'
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // max(1, num_envs),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=os.path.join(save_dir, 'tb')
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    final_path = os.path.join(save_dir, 'ppo_minitaur_obs_final')
    model.save(final_path)
    print(f"Saved final model to: {final_path}")


if __name__ == '__main__':
    main()
'''
'''
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
import time

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 3.0  # <--- CHANGED FROM 2.0 TO 3.0
        # Angles in radians: [-pi, -3pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3pi/4]
        # These correspond to Back, Back-left, Left, Front-left, Front, Front-right, Right, Back-right
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
            globalScaling=0.3
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
        sensor_height = 0.2
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
            if self._is_render:
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
        return self._get_observation_with_sensors(base_obs), reward, done, info

    def _get_forward_gait_action(self):
        t = self.step_counter / 20.0
        amplitude = 0.7
        frequency = 2.0
        hip_knee_fl_bl = amplitude * np.sin(2 * np.pi * frequency * t)
        hip_knee_fr_br = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)
        action = np.array([
            hip_knee_fl_bl, hip_knee_fl_bl,
            hip_knee_fr_br, hip_knee_fr_br,
            hip_knee_fl_bl, hip_knee_fl_bl,
            hip_knee_fr_br, hip_knee_fr_br
        ])
        return np.clip(action, -1.0, 1.0)

    def _get_turn_gait_action(self, direction="right"):
        t = self.step_counter / 20.0
        amplitude = 0.7
        frequency = 2.0
        turn_amp = 0.5
        
        hip_knee_fl_bl = amplitude * np.sin(2 * np.pi * frequency * t)
        hip_knee_fr_br = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)

        if direction == "right":
            fl_hip_offset, bl_hip_offset = turn_amp, turn_amp
            fr_hip_offset, br_hip_offset = -turn_amp, -turn_amp
        else: # "left"
            fl_hip_offset, bl_hip_offset = -turn_amp, -turn_amp
            fr_hip_offset, br_hip_offset = turn_amp, turn_amp

        action = np.array([
            hip_knee_fl_bl + fl_hip_offset, hip_knee_fl_bl,
            hip_knee_fr_br + fr_hip_offset, hip_knee_fr_br,
            hip_knee_fl_bl + bl_hip_offset, hip_knee_fl_bl,
            hip_knee_fr_br + br_hip_offset, hip_knee_fr_br
        ])
        return np.clip(action, -1.0, 1.0)

if __name__ == '__main__':
    env = MinitaurWithSensors(render=True)
    obs = env.reset()
    done = False

    print("Running: robot will walk forward and turn away if cube detected.")
    print("----------------------------------------------------------------")

    turning_steps = 0
    turn_direction = "left"
    
    # Store actions for better control
    forward_action = env._get_forward_gait_action()

    while not done:
        sensor_readings = obs[-8:]
        
        # CORRECTED SENSOR INDICES
        front_sensor = sensor_readings[4]
        left_sensor = sensor_readings[2]
        right_sensor = sensor_readings[6]
        
        obstacle_threshold = 0.8
        obstacle_detected = (
            front_sensor < obstacle_threshold or
            left_sensor < obstacle_threshold or
            right_sensor < obstacle_threshold
        )
        
        print(f"Front: {front_sensor:.2f}, Left: {left_sensor:.2f}, Right: {right_sensor:.2f} | Turning: {turning_steps > 0}")

        if obstacle_detected:
            if turning_steps == 0:
                turn_direction = "left" if left_sensor > right_sensor else "right"
                turning_steps = 200 # Increased turning steps for a clearer turn
                print(f"Obstacle detected! Initiating a {turn_direction} turn.")
            action = env._get_turn_gait_action(direction=turn_direction)
        elif turning_steps > 0:
            # Gradually blend the turning action back to the forward action
            turn_progress = (200 - turning_steps) / 200.0
            turn_action = env._get_turn_gait_action(direction=turn_direction)
            forward_action = env._get_forward_gait_action()
            action = turn_action * (1.0 - turn_progress) + forward_action * turn_progress
            
            turning_steps -= 1
            if turning_steps == 0:
                print("Finished turning, resuming forward gait.")
        else:
            action = env._get_forward_gait_action()

        obs, reward, done, _ = env.step(action)
        time.sleep(1.0 / 60)

    env.close()'''