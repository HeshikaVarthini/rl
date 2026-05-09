'''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PolicyNetworkWithObstacleAvoidance(nn.Module):
    """
    Enhanced Policy Network with obstacle avoidance capabilities
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, 
                 log_std_min=-20, log_std_max=2, device="cpu", 
                 num_sensors=8, obstacle_threshold=0.5):
        super(PolicyNetworkWithObstacleAvoidance, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        self.num_sensors = num_sensors
        self.obstacle_threshold = obstacle_threshold
        
        # Calculate original state dimension (without sensors)
        self.original_state_dim = state_dim - num_sensors
        
        # Original policy network
        self.linear1 = nn.Linear(self.original_state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        
        # Obstacle avoidance module
        self.obstacle_linear1 = nn.Linear(num_sensors, hidden_size // 2)
        self.obstacle_linear2 = nn.Linear(hidden_size // 2, action_dim)
        
        # Combine original features with obstacle avoidance features
        self.combine_linear = nn.Linear(hidden_size + action_dim, hidden_size)
        
        # Final output layers
        self.final_mean_linear = nn.Linear(hidden_size, action_dim)
        self.final_log_std_linear = nn.Linear(hidden_size, action_dim)
        
        # Move to device
        self.to(device)
    
    def forward(self, state):
        # Split state into original features and sensor readings
        original_features = state[:, :-self.num_sensors]
        sensor_readings = state[:, -self.num_sensors:]
        
        # Process original features
        x = F.relu(self.linear1(original_features))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        
        # Process sensor readings for obstacle avoidance
        obstacle_x = F.relu(self.obstacle_linear1(sensor_readings))
        obstacle_features = self.obstacle_linear2(obstacle_x)
        
        # Combine features
        combined = torch.cat([x, obstacle_features], dim=1)
        combined = F.relu(self.combine_linear(combined))
        
        # Final output
        final_mean = self.final_mean_linear(combined)
        final_log_std = torch.clamp(
            self.final_log_std_linear(combined), 
            self.log_std_min, 
            self.log_std_max
        )
        
        return final_mean, final_log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample action from distribution
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)
        action = torch.tanh(mean + std * z)
        
        # Calculate log probability
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        
        return action, log_prob
    
    def get_action(self, state, env=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # If environment is provided, use obstacle avoidance logic
        if env is not None and env.is_obstacle_ahead():
            # Get best direction from environment
            direction = env.get_best_direction()
            
            # Adjust action based on direction
            mean, log_std = self.forward(state_tensor)
            std = log_std.exp()
            
            # Sample from distribution
            normal = Normal(0, 1)
            z = normal.sample().to(self.device)
            action = torch.tanh(mean + std * z)
            
            # Modify action to turn in the best direction
            if direction == -1:  # Turn left
                action[0, 0] = max(action[0, 0], 0.5)  # Increase left turn
                action[0, 1] = min(action[0, 1], -0.5)  # Decrease right turn
            elif direction == 1:  # Turn right
                action[0, 0] = min(action[0, 0], -0.5)  # Decrease left turn
                action[0, 1] = max(action[0, 1], 0.5)  # Increase right turn
            
            return action.detach().cpu().numpy()[0]
        else:
            # Normal policy action
            mean, log_std = self.forward(state_tensor)
            std = log_std.exp()
            
            normal = Normal(0, 1)
            z = normal.sample().to(self.device)
            action = torch.tanh(mean + std * z)
            
            return action.detach().cpu().numpy()[0]'''
import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import os

class MinitaurWithSensors(gym.Env):
    """
    Custom PyBullet environment for Minitaur with obstacle avoidance.
    Adds fixed obstacles and simulated distance sensors.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render=False, num_sensors=8, sensor_range=1.5, max_steps=1000):
        super(MinitaurWithSensors, self).__init__()

        self.render_mode = render
        self.num_sensors = num_sensors
        self.sensor_range = sensor_range
        self.max_steps = max_steps
        self.step_count = 0

        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self._load_environment()
        self._load_robot()

        # Define observation and action spaces
        obs_dim = 28 + self.num_sensors  # 28 robot obs + 8 sensor readings
        act_dim = 8
        act_high = np.ones(act_dim)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _load_environment(self):
        """Loads ground and obstacles."""
        p.loadURDF("plane.urdf")

        # Use "cube_small.urdf" instead of missing "cube.urdf"
        self.obstacles = []
        obstacle_positions = [
            [1.0, 0.5, 0.1],
            [2.0, -0.5, 0.1],
            [3.0, 0.3, 0.1],
        ]
        for pos in obstacle_positions:
            obstacle_id = p.loadURDF("cube_small.urdf", pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.5)
            self.obstacles.append(obstacle_id)

    def _load_robot(self):
        """Load Minitaur robot."""
        self.robot = p.loadURDF("quadruped/minitaur.urdf", [0, 0, 0.2], useFixedBase=False)

    def _get_sensor_readings(self):
        """Return normalized distance sensor readings in 8 directions."""
        readings = []
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(base_ori)[2]

        for i in range(self.num_sensors):
            angle = yaw + (2 * np.pi * i / self.num_sensors)
            dx, dy = np.cos(angle), np.sin(angle)
            start = base_pos
            end = [base_pos[0] + dx * self.sensor_range, base_pos[1] + dy * self.sensor_range, base_pos[2]]
            result = p.rayTest(start, end)[0]
            hit_fraction = result[2] if result[0] != -1 else 1.0  # 1.0 = no hit
            readings.append(hit_fraction)

        return np.array(readings, dtype=np.float32)

    def _get_observation(self):
        """Combines robot observation with sensor readings."""
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        vel, ang_vel = p.getBaseVelocity(self.robot)
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))]

        obs = np.concatenate([
            np.array(pos),
            np.array(ori),
            np.array(vel),
            np.array(ang_vel),
            np.array(joint_states),
            self._get_sensor_readings()
        ])
        return obs.astype(np.float32)

    def step(self, action):
        """Applies action and returns obs, reward, done, info."""
        for i in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i], force=5)

        p.stepSimulation()
        self.step_count += 1

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.step_count >= self.max_steps

        return obs, reward, done, {}

    def _compute_reward(self):
        """Basic forward reward + obstacle avoidance penalty."""
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        reward = pos[0]  # reward for moving forward
        sensors = self._get_sensor_readings()
        penalty = np.sum(np.exp(-5 * sensors))  # exponential penalty for close obstacles
        return reward - penalty

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self._load_environment()
        self._load_robot()
        self.step_count = 0
        return self._get_observation()

    def render(self, mode='human'):
        pass  # GUI handled by pybullet

    def close(self):
        p.disconnect()

