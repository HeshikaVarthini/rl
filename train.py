from gym import spaces
import numpy as np
import pybullet as p

from sac import SAC2Agent, train_loop
from minisenchum1 import MinitaurWithSensors


class ActionBiasScaleWrapper:
    """
    Applies per-joint bias and global scaling to actions to stabilize gait.
    Keeps final actions within [-1, 1].
    """
    def __init__(self, env, knee_bias=0.3, scale=0.8):
        self.env = env
        self.knee_bias = float(knee_bias)
        self.scale = float(scale)

        assert isinstance(self.env.action_space, spaces.Box)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        num_joints = self.env.action_space.shape[0]
        bias = np.zeros(num_joints, dtype=np.float32)
        # Minitaur joint order: [FL_hip, FL_knee, FR_hip, FR_knee, BL_hip, BL_knee, BR_hip, BR_knee]
        knee_indices = [1, 3, 5, 7] if num_joints >= 8 else list(range(1, num_joints, 2))
        for idx in knee_indices:
            bias[idx] = self.knee_bias
        self._bias = bias

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.tanh(action)  # ensure in [-1,1]
        action = self.scale * action + self._bias
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)


class WalkingRewardWrapper:
    """
    Reward shaping for forward progress, posture, smoothness and obstacle avoidance.
    Uses robot base X delta, pitch/roll penalties, action smoothness, sensor proximity.
    """
    def __init__(self, env,
                 forward_gain=2.0,
                 posture_weight=1.0,
                 smooth_weight=0.05,
                 proximity_weight=3.0,
                 obstacle_threshold_m=0.8,
                 collision_cost=100.0,
                 alive_bonus=0.02):
        self.env = env
        self.forward_gain = float(forward_gain)
        self.posture_weight = float(posture_weight)
        self.smooth_weight = float(smooth_weight)
        self.proximity_weight = float(proximity_weight)
        self.obstacle_threshold_m = float(obstacle_threshold_m)
        self.collision_cost = float(collision_cost)
        self.alive_bonus = float(alive_bonus)

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._last_x = 0.0
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_x = self._get_base_x()
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)

        # Forward progress
        x = self._get_base_x()
        delta_x = x - self._last_x
        self._last_x = x
        r_forward = self.forward_gain * float(delta_x)

        # Posture penalty: pitch/roll away from level increases cost
        _, base_ori = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        roll, pitch, _ = p.getEulerFromQuaternion(base_ori)
        r_posture = -self.posture_weight * (abs(roll) + abs(pitch))

        # Smoothness penalty: discourage large action changes
        action_np = np.asarray(action, dtype=np.float32)
        r_smooth = -self.smooth_weight * float(np.linalg.norm(action_np - self._last_action, ord=1))
        self._last_action = action_np

        # Proximity penalty using front/left/right sensors from obs tail
        sensors = obs[-8:] if obs.shape[0] >= 8 else np.ones(8, dtype=np.float32)
        sensor_range = getattr(self.env, 'sensor_range', 2.0)
        front_m = float(sensors[4]) * sensor_range
        left_m = float(sensors[2]) * sensor_range
        right_m = float(sensors[6]) * sensor_range
        min_m = min(front_m, left_m, right_m)
        r_prox = -self.proximity_weight * max(0.0, self.obstacle_threshold_m - min_m)

        # Collision cost
        collided = False
        try:
            cps = p.getContactPoints(self.env.minitaur.quadruped, self.env.obstacle_id)
            collided = len(cps) > 0
        except Exception:
            pass
        r_collision = -self.collision_cost if collided else 0.0
        if collided:
            done = True

        reward = r_forward + r_posture + r_smooth + r_prox + r_collision + self.alive_bonus

        info = dict(info) if isinstance(info, dict) else {}
        info.update({
            "delta_x": delta_x,
            "r_forward": r_forward,
            "r_posture": r_posture,
            "r_smooth": r_smooth,
            "r_prox": r_prox,
            "r_collision": r_collision,
        })
        return obs, reward, done, info

    def _get_base_x(self):
        pos, _ = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        return float(pos[0])

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env(render=False):
    base = MinitaurWithSensors(render=render)
    wrapped = ActionBiasScaleWrapper(base, knee_bias=0.3, scale=0.8)
    wrapped = WalkingRewardWrapper(wrapped)
    return wrapped


if __name__ == "__main__":
    env = make_env(render=False)
    agent = SAC2Agent(env)
    rewards = train_loop(env, agent, max_total_steps=500000, max_steps=1000, batch_size=256, intermediate_policies=True, verbose=True)
    agent.save_policy("Policy1.pth")
    print("Training finished. Saved Policy1.pth")
