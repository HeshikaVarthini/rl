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
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Expect last 8 obs entries to be sensor fractions in [0,1]
        assert isinstance(self.observation_space, spaces.Box)
        assert self.observation_space.shape[0] >= 8, "Observation must include 8 sensor readings."

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_base_x = self._get_base_x()
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)

        # Forward progress along +X
        current_x = self._get_base_x()
        delta_x = current_x - self._last_base_x
        self._last_base_x = current_x

        forward_reward = self.forward_gain * float(delta_x)

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
    total_timesteps = int(os.environ.get('TOTAL_STEPS', str(300_000)))
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
