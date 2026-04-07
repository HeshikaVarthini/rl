import os
import time
import numpy as np

from stable_baselines3 import PPO

from minisenchum1 import MinitaurWithSensors
import gym

from train_minitaur_obstacle_rl import RewardShapingWrapper, ActionBiasScaleWrapper


def run(model_path: str, render: bool= True, episodes: int = 3):
    # Use same wrappers as training for consistency
    base_env = MinitaurWithSensors(render=render)
    gait_env = ActionBiasScaleWrapper(base_env, knee_bias=0.3, scale=0.8)
    env: gym.Env = RewardShapingWrapper(gait_env)

    model = PPO.load(model_path, env=None, print_system_info=True)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        steps = 0
        while not done and steps < 2500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rew += float(reward)
            steps += 1
            if render:
                time.sleep(1.0 / 120)

        print(f"Episode {ep+1}: reward={ep_rew:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    # Default path aligns with training script output
    default_model = os.path.join('checkpoints', 'ppo_minitaur_obstacle', 'ppo_minitaur_obs_final.zip')
    run(default_model, render=True, episodes=3)


