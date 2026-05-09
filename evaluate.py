'''from MinitaurWithSensors_ws import MinitaurWithSensors
from sac_ws import PolicyNetwork
import torch
import time
import pybullet as p

env = MinitaurWithSensors(render=True)
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

policy = PolicyNetwork(observation_dim, action_dim, hidden_dim, device=device)
policy.load_state_dict(torch.load("policy1.pth", map_location=device))
policy.eval()

rewards_all = []
for i in range(10):
    state = env.reset()
    rewards = []
    for _ in range(6): p.stepSimulation()
    for j in range(1000):
        action = policy.get_action(state)
        state, reward, done, _ = env.step(action)
        sensor_data = state[-8:]
        print(f"Episode {i+1}, Step {j+1}, Sensors: {sensor_data}")
        rewards.append(reward)
        if done: break
        time.sleep(0.03)
    rewards_all.append(sum(rewards))
    print(f"Episode {i+1}: Total reward = {sum(rewards)}")
print("Average reward:", sum(rewards_all) / 10)
env.close()'''
import time
from stable_baselines3 import PPO

# Import your environment
from MinitaurWithSensors_ws import MinitaurEnv   # change 'your_file' to your actual filename

# Load environment with rendering
env = MinitaurEnv(render=True)

# Load the trained PPO model
model = PPO.load("ppo_minitaur_obstacle", env=env)

# Run evaluation episodes
n_episodes = 5
for ep in range(n_episodes):
    obs = env.reset()
    done = False
    ep_reward = 0
    while not done:
        # Use trained policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ep_reward += reward
        time.sleep(1./60.)   # slow down for visualization
    print(f"Episode {ep+1}: total reward = {ep_reward:.2f}")

env.close()
