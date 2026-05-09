from minitaur_with_sensors import MinitaurWithSensors  # Use your custom env class
import pybullet as p
import torch
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv


env = MinitaurWithSensors(render=True)
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
