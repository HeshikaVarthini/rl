import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

from rlfinal1 import MinitaurObstacleEnv   # <-- your custom env file

def train():
    # Create env
    env = MinitaurObstacleEnv(render=False, num_obstacles=3, obstacle_type="mixed")

    # Check environment (works fine with Gym + SB3 < 2.0.0)
    check_env(env, warn=True)

    # Create model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_minitaur_tensorboard/")

    # Train model
    model.learn(total_timesteps=10000)

    # Save model
    model.save("ppo_minitaur_obstacles")
    env.close()
    print("✅ Training complete, model saved as ppo_minitaur_obstacles.zip")

def evaluate():
    # Reload environment & model
    env = MinitaurObstacleEnv(render=True, num_obstacles=3, obstacle_type="mixed")
    model = PPO.load("ppo_minitaur_obstacles", env=env)

    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True)
    print(f"✅ Evaluation -> mean_reward={mean_reward:.2f}, std={std_reward:.2f}")

    # Run one episode interactively
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(f"✅ Finished one demo episode, total_reward={total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train()
    evaluate()
