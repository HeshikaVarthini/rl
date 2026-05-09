import os
import numpy as np
import matplotlib.pyplot as plt
from sac_agent import SAC2Agent
from minitaur_obstacle_env import MinitaurObstacleEnv

def train_loop(env, agent, max_episodes=500, max_steps=1000, batch_size=256, 
               save_interval=50, eval_interval=10, verbose=True):
    """
    Training loop for obstacle avoidance
    """
    episode_rewards = []
    eval_rewards = []
    
    for episode in range(max_episodes):
        state = env.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            # Get action from policy
            if len(agent.replay_buffer) > batch_size:
                action = agent.policy.get_action(state)
            else:
                action = env.action_space.sample()
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, next_state, reward, done)
            
            # Update agent
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
            
            state = next_state
            ep_reward += reward
            
            if done:
                break
        
        episode_rewards.append(ep_reward)
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{max_episodes}, Reward: {ep_reward:.2f}, Avg Reward: {avg_reward:.2f}")
        
        # Save model
        if (episode + 1) % save_interval == 0:
            os.makedirs("./models", exist_ok=True)
            agent.save_policy(f"./models/minitaur_policy_episode_{episode+1}.pth")
            print(f"Model saved at episode {episode+1}")
        
        # Evaluate policy
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_policy(env, agent, max_steps)
            eval_rewards.append(eval_reward)
            print(f"Evaluation at episode {episode+1}: Reward = {eval_reward:.2f}")
    
    # Save final model
    os.makedirs("./models", exist_ok=True)
    agent.save_policy("./models/minitaur_final_policy.pth")
    print("Final model saved!")
    
    # Plot rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Evaluation Interval')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('./rewards_plot.png')
    plt.show()
    
    return episode_rewards, eval_rewards

def evaluate_policy(env, agent, max_steps, num_episodes=3):
    """Evaluate the current policy"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            action = agent.policy.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            ep_reward += reward
            
            if done:
                break
        
        total_rewards.append(ep_reward)
    
    return np.mean(total_rewards)

if __name__ == "__main__":
    # Create environment with obstacles
    env = MinitaurObstacleEnv(render=True, num_obstacles=10)
    
    # Create SAC2 agent
    agent = SAC2Agent(
        env=env,
        alpha=0.2,
        alr=3e-4,
        qlr=3e-4,
        policy_lr=3e-4,
        mem_size=1e6
    )
    
    # Train the agent
    train_loop(
        env=env,
        agent=agent,
        max_episodes=500,
        max_steps=1000,
        batch_size=256,
        save_interval=50,
        eval_interval=10,
        verbose=True
    )
    
    # Close environment
    env.close()