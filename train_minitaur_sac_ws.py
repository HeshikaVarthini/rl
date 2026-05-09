import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import random
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv


# -------------------------
# ENVIRONMENT
# -------------------------
class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 3.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.obstacle_id = None
        self.step_counter = 0
        self.sensor_lines = []
        self.last_x = 0.0  # for forward progress
        super().__init__(render=render)

        # Extend observation with sensors
        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(custom_obs_dim,), dtype=np.float32)

    def reset(self):
        self.step_counter = 0
        obs = super().reset()
        self.last_x = p.getBasePositionAndOrientation(self.minitaur.quadruped)[0][0]
        return self._get_observation_with_sensors(obs)

    def _get_observation_with_sensors(self, base_obs):
        sensor_data = []
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[1], base_matrix[2]])
        right_vector = np.array([base_matrix[3], base_matrix[4], base_matrix[5]])
        up_vector = np.array([base_matrix[6], base_matrix[7], base_matrix[8]])

        for line_id in self.sensor_lines:
            p.removeUserDebugItem(line_id)
        self.sensor_lines = []

        sensor_height = 0.3
        start = np.array(base_position) + sensor_height * up_vector
        for angle in self.sensor_angles:
            ray_dir = (
                np.cos(angle) * forward_vector +
                np.sin(angle) * right_vector
            )
            ray_dir /= np.linalg.norm(ray_dir)
            end = start + ray_dir * self.sensor_range
            result = p.rayTest(start, end)[0]
            hit_object_id = result[0]
            hit_fraction = result[2] if hit_object_id >= 0 else 1.0
            sensor_data.append(hit_fraction)

        return np.concatenate([base_obs, np.array(sensor_data)])

    def step(self, action):
        self.step_counter += 1
        base_obs, _, done, info = super().step(action)
        base_position, _ = p.getBasePositionAndOrientation(self.minitaur.quadruped)

        # --- Reward shaping ---
        reward = 0.0

        # Forward progress
        forward_progress = base_position[0] - self.last_x
        reward += 2.0 * forward_progress
        self.last_x = base_position[0]

        # Energy penalty (smoother walking)
        reward -= 0.1 * np.square(action).sum()

        # Fall penalty
        if base_position[2] < 0.18:
            reward -= 50.0
            done = True

        return self._get_observation_with_sensors(base_obs), reward, done, info


# -------------------------
# SAC COMPONENTS
# -------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        sample = map(np.stack, zip(*batch))
        return sample

    def __len__(self):
        return len(self.memory)


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, init_w=3e-3, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, init_w=3e-3,
                 log_std_min=-20, log_std_max=2, device="cpu"):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = torch.device(device)

        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.to(self.device)

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
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(self.device)
        action = torch.tanh(mean + std * z)
        return action.detach().cpu().numpy()[0]


class SAC2Agent:
    def __init__(self, env, gamma=0.99, tau=0.005, alpha=0.2,
                 policy_lr=3e-4, q_lr=3e-4, auto_alpha=True,
                 hidden_size=256, device="cpu"):
        self.device = torch.device(device)
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha

        self.replay_buffer = ReplayMemory(int(1e6))

        self.policy = PolicyNetwork(self.state_dim, self.action_dim, hidden_size, device=self.device)
        self.q_net1 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size, device=self.device)
        self.q_net2 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size, device=self.device)
        self.target_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size, device=self.device)
        self.target_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, hidden_size, device=self.device)

        for t, s in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            t.data.copy_(s.data)
        for t, s in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            t.data.copy_(s.data)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)

        if self.auto_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)
            self.alpha = self.log_alpha.exp()

    def update(self, batch_size):
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.evaluate(next_states)
            next_q1 = self.target_q_net1(next_states, next_actions)
            next_q2 = self.target_q_net2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * next_q

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

        actions_new, log_probs = self.policy.evaluate(states)
        q1 = self.q_net1(states, actions_new)
        q2 = self.q_net2(states, actions_new)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_probs - q).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for t, s in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)
        for t, s in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

    # --- save/load ---
    def save_policy(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()


# -------------------------
# TRAINING LOOP
# -------------------------
def train_loop(env, agent, max_total_steps, max_steps, batch_size):
    rewards = []
    steps = 0
    while steps < max_total_steps:
        state = env.reset()
        ep_reward = 0
        for step in range(max_steps):
            if steps > 2 * batch_size:
                action = agent.policy.get_action(state)
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, next_state, reward, done)
            state = next_state
            ep_reward += reward
            steps += 1

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode {len(rewards)}, reward={ep_reward:.2f}")
    return rewards


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    env = MinitaurWithSensors(render=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SAC2Agent(env, device=device)

    rewards = train_loop(env, agent, max_total_steps=200000, max_steps=500, batch_size=256)

    agent.save_policy("Policy1.pth")
    print("Policy saved!")
