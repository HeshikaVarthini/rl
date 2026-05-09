from sac import SAC2Agent, train_loop
from minitaur_with_sensors import MinitaurWithSensors
#from sac_with_obstacle_avoidance import PolicyNetworkWithObstacleAvoidance
import torch
import time
'''
# Create environment with sensors
env = MinitaurWithSensors(
    render=True, 
    drift_weight=0.625, 
    shake_weight=0.375, 
    energy_weight=0.005,
    num_sensors=8,
    sensor_range=1.0
)

# Create agent with modified policy network
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize agent with custom policy
agent = SAC2Agent(
    env,
    policy_network=PolicyNetworkWithObstacleAvoidance(
        observation_dim, 
        action_dim, 
        hidden_size=hidden_dim,
        device=device,
        num_sensors=8
    )
)

# Train the agent
train_loop(
    env, 
    agent, 
    max_total_steps=500000, 
    max_steps=1000, 
    batch_size=256, 
    intermediate_policies=True, 
    verbose=True
)

# Save final policy
agent.save_policy("quadruped_with_obstacle_avoidance.pth")'''
# training_and_sac_core.py
# This file contains the SAC agent, core networks, and the main training loop.

# Add this after network and environment class definitions
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sac_networks import GaussianPolicy, QNetwork, ValueNetwork
from replay_buffer import ReplayMemory

from minitaur_with_sensors import MinitaurWithSensors
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
gamma = 0.99
tau = 0.005
alpha = 0.2
lr = 3e-4
batch_size = 64
memory_capacity = 100000
num_episodes = 1000
start_steps = 1000
max_timesteps = 1000
update_every = 1

# Environment
env = MinitaurWithSensors(render=False, num_sensors=8, sensor_range=1.5)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Networks
policy = GaussianPolicy(state_dim, action_dim).to(device)
q1 = QNetwork(state_dim, action_dim).to(device)
q2 = QNetwork(state_dim, action_dim).to(device)
value = ValueNetwork(state_dim).to(device)
target_value = ValueNetwork(state_dim).to(device)
target_value.load_state_dict(value.state_dict())

# Optimizers
policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
q1_optimizer = optim.Adam(q1.parameters(), lr=lr)
q2_optimizer = optim.Adam(q2.parameters(), lr=lr)
value_optimizer = optim.Adam(value.parameters(), lr=lr)

# Replay buffer
memory = ReplayMemory(memory_capacity)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Training loop
for episode in range(num_episodes):
    state= env.reset()
    episode_reward = 0

    for t in range(max_timesteps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        if len(memory) < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = policy(state_tensor)
                action = action.cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push(state, action, next_state, reward, done)

        state = next_state
        episode_reward += reward

        if len(memory) >= batch_size:
            for _ in range(update_every):
                batch = memory.sample(batch_size)
                states, actions, next_states, rewards, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Update Q networks
                with torch.no_grad():
                    next_actions, next_log_probs = policy(next_states)
                    target_v = target_value(next_states)
                    q_target = rewards + (1 - dones) * gamma * target_v

                q1_loss = F.mse_loss(q1(states, actions), q_target)
                q2_loss = F.mse_loss(q2(states, actions), q_target)

                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                # Update Value network
                with torch.no_grad():
                    new_actions, log_probs = policy(states)
                    min_q = torch.min(q1(states, new_actions), q2(states, new_actions))
                    v_target = min_q - alpha * log_probs

                v_loss = F.mse_loss(value(states), v_target)
                value_optimizer.zero_grad()
                v_loss.backward()
                value_optimizer.step()

                # Update Policy
                new_actions, log_probs = policy(states)
                min_q = torch.min(q1(states, new_actions), q2(states, new_actions))
                policy_loss = (alpha * log_probs - min_q).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Soft update of target value network
                soft_update(target_value, value, tau)

        if done:
            break

    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward:.2f}")

env.close()
