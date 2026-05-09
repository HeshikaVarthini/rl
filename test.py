import pybullet as p
import torch
import time

from sac import PolicyNetwork
from train import ActionBiasScaleWrapper, WalkingRewardWrapper
from minisenchum1 import MinitaurWithSensors

env = WalkingRewardWrapper(ActionBiasScaleWrapper(MinitaurWithSensors(render=True), knee_bias=0.3, scale=0.8))

observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

policy = PolicyNetwork(observation_dim, action_dim, hidden_dim, device=device)
policy.load_state_dict(torch.load("./Policy1.pth", map_location=device))
policy.eval()

r = []
for i in range(10):
    state = env.reset()
    rewards = []
    for _ in range(6):
        p.stepSimulation()
    for j in range(1000):
        action = policy.get_action(state)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
        time.sleep(0.03)
    r.append(sum(rewards))
    print(f"Episode {i+1}: Total reward = {sum(rewards)}")

print("Average reward: ", sum(r)/len(r))