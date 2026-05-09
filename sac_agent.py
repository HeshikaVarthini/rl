import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import random
import numpy as np

class ReplayMemory(object):
    """
    Finite list object for experience replay
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

class PolicyNetwork(nn.Module):
    """
    Gaussian Policy network
    """
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3, 
                 log_std_min=-20, log_std_max=2, device="cpu"):
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

class SAC2Agent:
    """
    Agent for the second generation of the Soft Actor Critic learning algorithm
    """
    def __init__(self, env, alpha=0.2, alr=3e-4, qlr=3e-4, policy_lr=3e-4, mem_size=1e6):
        try:
            action_dim = env.action_space.shape[0]
        except IndexError:
            action_dim = env.action_space.n
        try:
            observation_dim = env.observation_space.shape[0]
        except IndexError:
            observation_dim = env.observation_space.n
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
        self.q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)
        self.target_q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
        self.target_q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        self.policy = PolicyNetwork(observation_dim, action_dim, 256, device=self.device).to(self.device)
        
        self.alpha = alpha
        self.target_a = -action_dim
        self.log_a = torch.zeros(1, requires_grad=True, device=self.device)
        
        self.q1_criterion = nn.MSELoss().to(self.device)
        self.q2_criterion = nn.MSELoss().to(self.device)
        
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=qlr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=qlr)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.a_optim = optim.Adam([self.log_a], lr=alr)
        
        self.mem_size = mem_size
        self.replay_buffer = ReplayMemory(mem_size)
        
    def update(self, batch_size, gamma=0.99, tau=5e-3):
        """
        Update the parameters of the agent
        """
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
        
        # Convert all to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        
        next_action, next_log_prob = self.policy.evaluate(next_state)
        
        # Update Q networks using the loss function
        next_q1 = self.target_q1(next_state, next_action)
        next_q2 = self.target_q2(next_state, next_action)
        value = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
        expected_q = reward + gamma * (1 - done) * value
        
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        q1_loss = self.q1_criterion(q1, expected_q.detach())
        q2_loss = self.q2_criterion(q2, expected_q.detach())
        
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()
        
        # Update policy network with loss function
        new_action, log_prob = self.policy.evaluate(state)
        policy_loss = (self.alpha * log_prob - torch.min(self.q1(state, new_action), self.q2(state, new_action))).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # Update temperature with loss function
        alpha_loss = (self.log_a * (-log_prob - self.target_a).detach()).mean()
        
        self.a_optim.zero_grad()
        alpha_loss.backward()
        self.a_optim.step()
        self.alpha = self.log_a.exp()
        
        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    def save_policy(self, path):
        """
        Saves the state dictionary of the policy network to the specified path
        """
        torch.save(self.policy.state_dict(), path)