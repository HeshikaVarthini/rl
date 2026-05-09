import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

from rl.ppo_agent import PPOAgent, PPOBuffer
from rl.rl_environment import QuadrupedEnv
from utils.logger import LoggerManager

class RLTrainer:
    """Reinforcement learning trainer class for quadruped robots"""
    
    def __init__(self, config_file=None, render=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 console_output=True):
        """
        Initialize the trainer
        
        Args:
            config_file (str): Path to configuration file
            render (bool): Whether to render during evaluation
            device (str): Device to use (cuda/cpu)
            console_output (bool): Whether to output logs to console
        """
        self.config_file = config_file
        self.render = render
        self.device = device
        self.console_output = console_output
        
        # Default parameters
        self.hyperparams = {
            'hidden_dim': 128,
            'lr': 3e-4,
            'gamma': 0.99,
            'lam': 0.95,
            'clip_ratio': 0.2,
            'target_kl': 0.01,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'update_epochs': 4,
            'action_update_freq': 5,
            'max_steps': 1000,
            'steps_per_epoch': 2000,
            'save_freq': 10,
            'eval_episodes': 5,
            'eval_delay': 0.01
        }
        
        # Initialize timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Initialize logger
        self.log_dir = os.path.join("logs", f"rl_{self.timestamp}")
        self.results_dir = os.path.join("results", f"ppo_quadruped_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger_manager = LoggerManager(log_dir=self.log_dir)
        
        # Get various loggers
        self.main_logger = self.logger_manager.get_logger('main', console_output=console_output)
        self.train_logger = self.logger_manager.get_logger('train', console_output=console_output)
        self.eval_logger = self.logger_manager.get_logger('eval', console_output=console_output)
        self.env_logger = self.logger_manager.get_logger('environment', console_output=False)
        self.agent_logger = self.logger_manager.get_logger('agent', console_output=False)
        
        self.log_info(f"Initialized RL trainer - Device: {device}, Config file: {config_file}")
        self.log_info(f"Log files will be saved to {self.log_dir}")
        self.log_info(f"Results will be saved to {self.results_dir}")
    
    def log_info(self, message):
        """Output information log (both to file and console)"""
        self.main_logger.info(message)
        if self.console_output:
            print(message)
    
    def _plot_training_metrics(self, metrics, save_path=None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot rewards
        axes[0, 0].plot(metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Plot episode lengths
        axes[0, 1].plot(metrics['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Plot policy loss
        axes[1, 0].plot(metrics['policy_losses'])
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        
        # Plot value function loss
        axes[1, 1].plot(metrics['value_losses'])
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.log_info(f"Learning curves saved to {save_path}")
        
        # Close figure (prevent memory leak)
        plt.close(fig)
    
    def _save_metrics(self, metrics, path):
        """Save metrics as a JSON file"""
        # Convert metrics to JSON format
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                # Convert NumPy arrays to lists
                json_metrics[key] = [v.tolist() for v in value]
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.number):
                # Convert NumPy numbers to lists
                json_metrics[key] = [float(v) for v in value]
            elif isinstance(value, list):
                json_metrics[key] = value
            elif isinstance(value, np.number):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        
        # Save as JSON file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2)
        
        self.log_info(f"Metrics saved to {path}")
    
    def train(self, epochs=100, load_model=None):
        """Train an agent using the PPO algorithm"""
        self.log_info(f"Starting training - Number of epochs: {epochs}")
        print(f"Starting training - Number of epochs: {epochs}")
        
        # Initialize environment
        env = QuadrupedEnv(
            config_file=self.config_file, 
            render=self.render, 
            max_steps=self.hyperparams['max_steps'],
            logger=self.env_logger
        )
        
        # Get action space dimension and range
        action_dim = env.action_space.shape[0]
        action_bounds = (env.action_space.low, env.action_space.high)
        
        # Get state space dimension
        state_dim = env.observation_space.shape[0]
        
        # Initialize PPO agent
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            hidden_dim=self.hyperparams['hidden_dim'],
            lr=self.hyperparams['lr'],
            gamma=self.hyperparams['gamma'],
            lam=self.hyperparams['lam'],
            clip_ratio=self.hyperparams['clip_ratio'],
            target_kl=self.hyperparams['target_kl'],
            value_coef=self.hyperparams['value_coef'],
            entropy_coef=self.hyperparams['entropy_coef'],
            update_epochs=self.hyperparams['update_epochs'],
            action_update_freq=self.hyperparams['action_update_freq'],
            device=self.device,
            logger=self.agent_logger
        )
        
        # Load existing model
        if load_model:
            agent.load_model(load_model)
            self.log_info(f"Loaded model from {load_model}")
            print(f"Loaded model from {load_model}")
        
        # Calculate buffer size
        buffer_size = self.hyperparams['steps_per_epoch']
        
        # Initialize buffer
        buffer = PPOBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            gamma=self.hyperparams['gamma'],
            lam=self.hyperparams['lam']
        )
        
        # Training loop
        self.log_info(f"Starting training loop")
        print(f"Starting training loop")
        start_time = time.time()
        
        # Initialize
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_num = 0
        
        # For storing metrics
        episode_rewards = []
        episode_lengths = []
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Data collection phase
            self.train_logger.info(f"Starting data collection for epoch {epoch+1}/{epochs}")
            if self.console_output:
                print(f"Starting data collection for epoch {epoch+1}/{epochs}")
            
            for t in range(buffer_size):
                # Select action
                denorm_action, log_prob, value = agent.select_action(state)
                
                # Normalize action
                norm_action = np.copy(denorm_action)
                for i in range(action_dim):
                    low, high = env.action_space.low[i], env.action_space.high[i]
                    norm_action[i] = 2.0 * (denorm_action[i] - low) / (high - low) - 1.0
                
                # Execute action in environment
                next_state, reward, done, info = env.step(denorm_action)
                
                # Store in buffer
                buffer.store(state, norm_action, reward, value, log_prob, done)
                
                # Update state
                state = next_state
                
                # Update episode statistics
                episode_reward += reward
                episode_length += 1
                
                # Handle episode completion
                if done:
                    # Record episode statistics
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    agent.metrics['episode_rewards'].append(episode_reward)
                    agent.metrics['episode_lengths'].append(episode_length)
                    
                    # Log to file
                    episode_info = {
                        'episode': episode_num + 1,
                        'reward': episode_reward,
                        'length': episode_length,
                        'info': info
                    }
                    self.train_logger.info(f"Episode completed: {episode_info}")
                    
                    # Also output to console
                    if self.console_output:
                        print(f"Episode {episode_num+1} completed: reward={episode_reward:.2f}, length={episode_length}")
                    
                    # Start new episode
                    state = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    episode_num += 1
            
            # Get value of last state (if episode not done)
            if not done:
                _, _, last_value = agent.select_action(state)
            else:
                last_value = 0
            
            # Calculate advantages and returns
            buffer.compute_advantages(last_value)
            
            # Update policy
            self.train_logger.info(f"Starting policy update for epoch {epoch+1}/{epochs}")
            if self.console_output:
                print(f"Starting policy update for epoch {epoch+1}/{epochs}")
            
            update_info = agent.update(buffer)
            
            # Log update information
            self.train_logger.info(f"Policy update completed: {update_info}")
            
            # Clear buffer
            buffer.clear()
            
            # Record epoch end time
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            epoch_summary = f"Epoch {epoch+1}/{epochs} completed - Duration: {epoch_duration:.2f} seconds"
            self.train_logger.info(epoch_summary)
            if self.console_output:
                print(epoch_summary)
                print(f"  Policy loss: {update_info['policy_loss']:.6f}, Value loss: {update_info['value_loss']:.6f}")
                print(f"  KL: {update_info['kl_divergence']:.6f}, Entropy: {update_info['entropy']:.6f}")
            
            # Periodic model saving
            if (epoch + 1) % self.hyperparams['save_freq'] == 0 or epoch == epochs - 1:
                model_path = os.path.join(self.results_dir, f"model_epoch_{epoch+1}.pt")
                agent.save_model(model_path)
                
                # Save metrics
                metrics_path = os.path.join(self.results_dir, f"metrics_epoch_{epoch+1}.json")
                self._save_metrics(agent.metrics, metrics_path)
                
                # Plot learning curves
                plot_path = os.path.join(self.results_dir, f"learning_curves_epoch_{epoch+1}.png")
                self._plot_training_metrics(agent.metrics, save_path=plot_path)
                
                # Console output
                if self.console_output:
                    print(f"Saved model and metrics (epoch {epoch+1})")
        
        # Training end time
        total_time = time.time() - start_time
        self.log_info(f"Training completed. Total time: {total_time:.2f} seconds")
        print(f"Training completed. Total time: {total_time:.2f} seconds")
        
        # Close environment
        env.close()
        
        # Save final model
        final_model_path = os.path.join(self.results_dir, "model_final.pt")
        agent.save_model(final_model_path)
        
        # Save final metrics
        final_metrics_path = os.path.join(self.results_dir, "metrics_final.json")
        self._save_metrics(agent.metrics, final_metrics_path)
        
        # Plot final learning curves
        final_plot_path = os.path.join(self.results_dir, "learning_curves_final.png")
        self._plot_training_metrics(agent.metrics, save_path=final_plot_path)
        
        self.log_info(f"Final model saved to {final_model_path}")
        print(f"Final model saved to {final_model_path}")
        
        # Save hyperparameters
        hyperparams_path = os.path.join(self.results_dir, "hyperparameters.json")
        with open(hyperparams_path, 'w', encoding='utf-8') as f:
            json.dump(self.hyperparams, f, indent=2)
            
        self.log_info(f"Hyperparameters saved to {hyperparams_path}")
        
        return final_model_path
    
    def evaluate(self, model_path, episodes=5, delay=0.01):
        """Evaluate a trained agent"""
        self.log_info(f"Starting evaluation of model {model_path} - Number of episodes: {episodes}")
        print(f"Starting evaluation of model {model_path} - Number of episodes: {episodes}")
        
        # Initialize environment
        env = QuadrupedEnv(
            config_file=self.config_file, 
            render=True, 
            max_steps=self.hyperparams['max_steps'],
            logger=self.env_logger
        )
        
        # Get action space dimension and range
        action_dim = env.action_space.shape[0]
        action_bounds = (env.action_space.low, env.action_space.high)
        
        # Get state space dimension
        state_dim = env.observation_space.shape[0]
        
        # Initialize PPO agent
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            hidden_dim=self.hyperparams['hidden_dim'],
            action_update_freq=self.hyperparams['action_update_freq'],
            device=self.device,
            logger=self.agent_logger
        )
        
        # Load model
        agent.load_model(model_path)
        self.log_info(f"Loaded model from {model_path}")
        print(f"Loaded model from {model_path}")
        
        # Arrays for storing evaluation results
        total_rewards = []
        total_lengths = []
        success_count = 0
        
        # Evaluation loop
        for episode in range(episodes):
            self.eval_logger.info(f"Starting evaluation episode {episode+1}/{episodes}")
            if self.console_output:
                print(f"Starting evaluation episode {episode+1}/{episodes}")
            
            state = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            
            while not done:
                # Deterministic action selection
                denorm_action, _, _ = agent.select_action(state, deterministic=True)
                episode_actions.append(denorm_action.tolist())
                
                # Execute action in environment
                next_state, reward, done, info = env.step(denorm_action)
                
                # Update state
                state = next_state
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                
                # Log step information
                if episode_length % 100 == 0:
                    step_msg = f"Episode {episode+1}, Step {episode_length}, Current reward {episode_reward:.2f}"
                    self.eval_logger.info(step_msg)
                    if self.console_output:
                        print(step_msg)
                
                # Adjust execution speed (for better visual understanding)
                if delay > 0:
                    time.sleep(delay)
            
            # Record episode results
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            
            # Check if goal was reached
            if 'goal_reached' in info and info['goal_reached']:
                success_count += 1
                goal_msg = f"Episode {episode+1} reached the goal!"
                self.eval_logger.info(goal_msg)
                if self.console_output:
                    print(goal_msg)
            
            # Log episode information
            episode_info = {
                'episode': episode + 1,
                'reward': episode_reward,
                'length': episode_length,
                'success': 'goal_reached' in info and info['goal_reached'],
                'info': info
            }
            self.eval_logger.info(f"Evaluation episode completed: {episode_info}")
            if self.console_output:
                print(f"Evaluation episode {episode+1} completed: reward={episode_reward:.2f}, length={episode_length}")
            
            # Save episode action records
            actions_path = os.path.join(self.results_dir, f"eval_episode_{episode+1}_actions.json")
            with open(actions_path, 'w', encoding='utf-8') as f:
                json.dump(episode_actions, f)
        
        # Calculate evaluation summary
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(total_lengths)
        success_rate = success_count / episodes * 100
        
        # Log results
        summary = {
            'episodes': episodes,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'success_rate': success_rate,
            'success_count': success_count
        }
        self.log_info(f"Evaluation completed: {summary}")
        
        # Display results to console
        if self.console_output:
            print("\nEvaluation Summary:")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Average episode length: {avg_length:.2f}")
            print(f"  Success rate: {success_rate:.2f}% ({success_count}/{episodes})")
        
        # Save evaluation results to file
        eval_results_path = os.path.join(self.results_dir, "evaluation_results.json")
        results = {
            'model_path': model_path,
            'episodes': episodes,
            'rewards': [float(r) for r in total_rewards],
            'lengths': total_lengths,
            'success_rate': float(success_rate),
            'success_count': success_count,
            'avg_reward': float(avg_reward),
            'avg_length': float(avg_length)
        }
        
        with open(eval_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        self.log_info(f"Evaluation results saved to {eval_results_path}")
        print(f"Evaluation results saved to {eval_results_path}")
        
        # Close environment
        env.close()
        
        return avg_reward, success_rate