from minitaur_with_sensors import MinitaurWithSensors
from sac_with_obstacle_avoidance import PolicyNetworkWithObstacleAvoidance
import pybullet as p
import torch
import time

# Create environment with sensors
env = MinitaurWithSensors(render=True, num_sensors=8, sensor_range=1.0)

# Get dimensions
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create policy network
policy = PolicyNetworkWithObstacleAvoidance(
    observation_dim, 
    action_dim, 
    hidden_size=hidden_dim,
    device=device,
    num_sensors=8
)

# Load trained policy
policy.load_state_dict(torch.load("./quadruped_with_obstacle_avoidance.pth", map_location=device))
policy.eval()  # Set to evaluation mode

# Test the policy
rewards = []
for i in range(10):
    state = env.reset()
    episode_rewards = []
    
    # Let minitaur land
    for _ in range(6):
        p.stepSimulation()
    
    # Run episode
    for j in range(1000):
        # Get action from policy with obstacle avoidance
        action = policy.get_action(state, env=env)
        state, reward, _, _ = env.step(action)
        episode_rewards.append(reward)
        
        if env.is_fallen():
            break
            
        time.sleep(0.03)
    
    total_reward = sum(episode_rewards)
    rewards.append(total_reward)
    print(f"Episode {i+1}: Total reward = {total_reward}")

print("Average reward: ", sum(rewards)/10)