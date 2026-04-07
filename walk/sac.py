import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import random
import numpy as np
import pybullet as p

class ReplayMemory(object):
    """
    Finite list object
    """
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0
        
    def push(self, state, action, next_state, reward, done):
        """
        Save a state transition to the memory
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Returns a random sample of the memory of size batch_size
        """
        if batch_size >= self.capacity:
            batch = self.memory
        else:
            batch = random.sample(self.memory, batch_size)
        sample = map(np.stack, zip(*batch))
        return sample
    
    def __len__(self):
        return len(self.memory)

class SoftQNetwork(nn.Module):
    """
    Soft Q Neural Network
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ValueNetwork(nn.Module):
    """
    Value Network
    Only used in SACAgent, replaced by entropy temperature in SAC2Agent
    """
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    """
    Gaussian Policy network
    """
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, device="cpu"):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
        # Move the entire network to the specified device
        self.to(device)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # Sample an action from the gaussian distribution with the mean and std
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)  # Ensure z is on the same device
        action = torch.tanh(mean + std * z)
        # Get the log of the probability of action plus some noise
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob
    
    def get_action(self, state):
        # Convert state to tensor and move to device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # Sample an action from the gaussian distribution with the mean and std
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)  # Ensure z is on the same device
        action = torch.tanh(mean + std * z)
        # Move action to CPU and convert to numpy
        return action.detach().cpu().numpy()[0]

# Rest of the SACAgent and SAC2Agent classes remain the same
# ...

def train_loop(env, agent, max_total_steps, max_steps, batch_size, intermediate_policies=False, path="./", verbose=False, update_all=True):
    """
    Training loop
    """
    rewards = []
    steps = 0
    while steps < max_total_steps:
        state = env.reset()
        ep_reward = 0
        # Step the simulation 5 to stop learning starting midair if it is the minitaur env
        try:
            env.minitaur
            for i in range(5):
                p.stepSimulation()
        except AttributeError:
            continue
        for step in range(max_steps):
            if verbose and not (steps % (max_total_steps // 100)):
                print("Steps: {}".format(steps))
            # Get random action until the replay memory has been filled, then get action from policy network
            if steps > 2 * batch_size:
                action = agent.policy.get_action(state)
                next_state, reward, done, _ = env.step(action)
            else:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
            # Add state action transition to replay memory
            agent.replay_buffer.push(state, action, next_state, reward, done)
            state = next_state
            ep_reward += reward
            steps += 1
            if update_all:
                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)
            else:
                if len(agent.replay_buffer) > batch_size and not steps % 10:
                    agent.update(batch_size)
            # Save the policy network at 20% increments
            if intermediate_policies and not steps % (max_total_steps // 5):
                agent.save_policy(path + "policy{}.pth".format((steps // (max_total_steps // 5))))
            # Break out of loop if an end state has been reached
            if done:
                break
        rewards.append(ep_reward)
    return rewards
# Add this to your sac.py file

class SAC2Agent:
    """
    Soft Actor-Critic Agent with automatic temperature adjustment
    """
    def __init__(self, env, gamma=0.99, tau=0.005, alpha=0.2, 
                 policy_lr=3e-4, q_lr=3e-4, auto_alpha=True, 
                 policy_network=None, hidden_size=256):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        
        # Create replay buffer
        self.replay_buffer = ReplayMemory(int(1e6))
        
        # Create policy network
        if policy_network is None:
            self.policy = PolicyNetwork(self.state_dim, self.action_dim, hidden_size)
        else:
            self.policy = policy_network
            
        # Create Q networks
        self.q_net1 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size)
        self.q_net2 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size)
        self.target_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size)
        self.target_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size)
        
        # Initialize target networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param.data)
        
        # Create optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        
        # Automatic temperature adjustment
        if self.auto_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.q_net1.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.q_net1.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)
            self.alpha = self.log_alpha.exp()
    
    def update(self, batch_size):
        # Sample from replay buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.q_net1.device)
        actions = torch.FloatTensor(actions).to(self.q_net1.device)
        next_states = torch.FloatTensor(next_states).to(self.q_net1.device)
        rewards = torch.FloatTensor(rewards).to(self.q_net1.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.q_net1.device).unsqueeze(1)
        
        # Compute Q targets
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.evaluate(next_states)
            next_q1 = self.target_q_net1(next_states, next_actions)
            next_q2 = self.target_q_net2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * next_q
        
        # Update Q networks
        current_q1 = self.q_net1(states, actions)
        current_q2 = self.q_net2(states, actions)
        
        q1_loss = F.mse_loss(current_q1, q_target)
        q2_loss = F.mse_loss(current_q2, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        actions, log_probs = self.policy.evaluate(states)
        q1 = self.q_net1(states, actions)
        q2 = self.q_net2(states, actions)
        q = torch.min(q1, q2)
        
        policy_loss = (self.alpha * log_probs - q).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update temperature
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save_policy(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path))
