import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import os
import logging

class ActorCritic(nn.Module):
    """
    Model containing pyramid-style actor and critic networks
    - Layer width gradually narrows (for recognition problems)
    """
    def __init__(self, state_dim, action_dim, base_dim=128):
        super(ActorCritic, self).__init__()
        
        # Set pyramid-style widths
        # Gradually narrows from input to output
        self.layer1_dim = base_dim
        self.layer2_dim = base_dim // 2
        self.layer3_dim = base_dim // 4
        
        # Actor network (state -> action mean and standard deviation)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, self.layer1_dim),
            nn.ReLU(),
            nn.Linear(self.layer1_dim, self.layer2_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_dim, self.layer3_dim),
            nn.ReLU()
        )
        
        # Layer for outputting action mean
        self.mean_layer = nn.Linear(self.layer3_dim, action_dim)
        
        # Layer for outputting action standard deviation (as learnable parameters)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (state -> state value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.layer1_dim),
            nn.ReLU(),
            nn.Linear(self.layer1_dim, self.layer2_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_dim, self.layer3_dim),
            nn.ReLU(),
            nn.Linear(self.layer3_dim, 1)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            # Apply He initialization (suitable for ReLU activation function)
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, state):
        """
        Takes state as input and returns action distribution and state value
        """
        # Calculate actor output
        actor_hidden = self.actor(state)
        action_mean = self.mean_layer(actor_hidden)
        
        # Calculate action standard deviation (always positive)
        action_std = torch.exp(self.log_std)
        
        # Calculate state value
        value = self.critic(state)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """
        Takes state as input and samples an action
        If deterministic=True, returns the mean value of the action
        """
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            # Deterministic action (used during evaluation)
            return action_mean
        else:
            # Stochastic action (used during training)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            return action
    
    def evaluate(self, state, action):
        """
        Takes state and action as input and calculates log probability, entropy, and state value
        """
        action_mean, action_std, value = self.forward(state)
        
        # Action probability distribution
        dist = Normal(action_mean, action_std)
        
        # Log probability of action
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Distribution entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value

class PPOBuffer:
    """
    Buffer used in PPO algorithm
    Stores episode data and samples mini-batches during training
    """
    def __init__(self, state_dim, action_dim, buffer_size, gamma=0.99, lam=0.95):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.buffer_size = buffer_size
    
    def store(self, state, action, reward, value, log_prob, done):
        """
        Store single step data in the buffer
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
    
    def compute_advantages(self, last_value=0.0):
        """
        Compute advantages and returns using GAE (Generalized Advantage Estimation)
        """
        path_slice = slice(0, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        dones = np.append(self.dones[path_slice], 0)
        
        # GAE calculation
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Return calculation
        self.returns[path_slice] = self._discount_cumsum(rewards[:-1], self.gamma)
    
    def _discount_cumsum(self, x, discount):
        """
        Calculate discounted cumulative sum
        x: [x0, x1, x2, ...]
        returns: [x0 + discount*x1 + discount^2*x2 + ..., x1 + discount*x2 + ...]
        """
        discounted_sum = np.zeros_like(x)
        running_sum = 0
        for i in reversed(range(len(x))):
            running_sum = x[i] + discount * running_sum * (1 - self.dones[i])
            discounted_sum[i] = running_sum
        return discounted_sum
    
    def get_batches(self, batch_size=64):
        """
        Generate batch data from buffer
        """
        path_slice = slice(0, self.ptr)
        indices = np.random.permutation(self.ptr)
        
        for i in range(0, self.ptr, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield {
                'states': self.states[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'returns': self.returns[batch_indices],
                'advantages': self.advantages[batch_indices]
            }
    
    def clear(self):
        """
        Clear the buffer
        """
        self.ptr = 0

class PPOAgent:
    """
    Agent implementing the PPO (Proximal Policy Optimization) algorithm
    """
    def __init__(self, state_dim, action_dim, action_bounds, hidden_dim=64, lr=3e-4, 
                 gamma=0.99, lam=0.95, clip_ratio=0.2, target_kl=0.01, 
                 value_coef=0.5, entropy_coef=0.01, update_epochs=4, 
                 action_update_freq=5, device='cuda' if torch.cuda.is_available() else 'cpu',
                 logger=None):
        """
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            action_bounds: Action range tuple (min, max)
            hidden_dim: Number of units in hidden layers
            lr: Learning rate
            gamma: Discount factor
            lam: GAE lambda parameter
            clip_ratio: PPO clipping parameter
            target_kl: Target KL divergence
            value_coef: Value function loss coefficient
            entropy_coef: Entropy regularization coefficient
            update_epochs: Number of updates per batch
            action_update_freq: Action update frequency (for gait stability) - reduced from 10 to 5 to increase update frequency
            device: Device used for computation
            logger: Logger instance
        """
        # Logger setup
        self.logger = logger or logging.getLogger('ppo_agent')
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.action_update_freq = action_update_freq
        
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        
        # Initialize model
        self.model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Action counter (for gait stability)
        self.action_counter = 0
        self.current_action = np.zeros(action_dim)
        
        # Training metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy': [],
            'kl': []
        }
        
        self.logger.info(f"PPO agent initialized: state_dim={state_dim}, action_dim={action_dim}, device={device}")
        self.logger.info(f"Action range: {action_bounds}")
        self.logger.info(f"Hyperparameters: gamma={gamma}, lam={lam}, clip_ratio={clip_ratio}, lr={lr}")
        self.logger.info(f"Action update frequency: {action_update_freq} (smaller means more frequent updates)")
    
    def _normalize_action(self, action):
        """
        Convert neural network output to actual action range
        """
        low, high = self.action_bounds
        action = np.clip(action, -1.0, 1.0)
        action = low + (action + 1.0) * 0.5 * (high - low)
        return np.clip(action, low, high)
    
    def _denormalize_action(self, action):
        """
        Convert actual action to neural network input range
        """
        low, high = self.action_bounds
        action = np.clip(action, low, high)
        action = 2.0 * (action - low) / (high - low) - 1.0
        return np.clip(action, -1.0, 1.0)
    
    def select_action(self, state, deterministic=False):
        """
        Select action based on state
        Limit action update frequency for gait stability
        """
        if self.action_counter % self.action_update_freq == 0:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if deterministic:
                    action = self.model.get_action(state_tensor, deterministic=True)
                else:
                    action = self.model.get_action(state_tensor)
                    
                    # Get log probability and state value
                    log_prob, _, value = self.model.evaluate(state_tensor, action)
            
            # Convert tensors to numpy arrays
            action = action.cpu().numpy().flatten()
            log_prob = log_prob.cpu().numpy().flatten()[0] if not deterministic else 0
            value = value.cpu().numpy().flatten()[0] if not deterministic else 0
            
            # Convert to actual action range
            denorm_action = self._normalize_action(action)
            
            # Update current action
            self.current_action = denorm_action
            
            # Log the action
            action_str = ", ".join([f"{a:.3f}" for a in denorm_action])
            self.logger.debug(f"Selected new action: [{action_str}], prob={log_prob:.3f}, value={value:.3f}")
        else:
            # Return the same action as before (for gait stability)
            action = self._denormalize_action(self.current_action)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get log probability and state value
                log_prob, _, value = self.model.evaluate(state_tensor, action_tensor)
            
            log_prob = log_prob.cpu().numpy().flatten()[0] if not deterministic else 0
            value = value.cpu().numpy().flatten()[0] if not deterministic else 0
            denorm_action = self.current_action
            
            # Log (low level)
            self.logger.debug(f"Maintaining action: [{', '.join([f'{a:.3f}' for a in denorm_action])}]")
        
        # Update action counter
        self.action_counter += 1
        
        return denorm_action, log_prob, value
    
    def update(self, buffer):
        """
        Update policy and value function using collected data
        """
        # Calculate advantages and returns
        buffer.compute_advantages()
        
        # Variables for recording metrics
        value_losses = []
        policy_losses = []
        entropy_terms = []
        kl_divs = []
        
        self.logger.info(f"Starting policy update: buffer size={buffer.ptr}, update epochs={self.update_epochs}")
        
        # Update in each epoch
        for epoch in range(self.update_epochs):
            epoch_start_time = time.time()
            epoch_value_losses = []
            epoch_policy_losses = []
            epoch_entropy_terms = []
            epoch_kl_divs = []
            
            batch_count = 0
            for batch in buffer.get_batches():
                # Convert batch data to tensors
                states = torch.FloatTensor(batch['states']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)
                old_log_probs = torch.FloatTensor(batch['log_probs']).unsqueeze(1).to(self.device)
                returns = torch.FloatTensor(batch['returns']).unsqueeze(1).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).unsqueeze(1).to(self.device)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Evaluate actions with current policy
                new_log_probs, entropy, values = self.model.evaluate(states, actions)
                
                # Calculate policy ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Calculate PPO objective function
                obj = ratio * advantages
                obj_clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()
                
                # Calculate value function loss
                value_loss = ((values - returns) ** 2).mean()
                
                # Calculate entropy
                entropy_term = entropy.mean()
                
                # Calculate total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_term
                
                # Calculate KL divergence
                approx_kl = ((old_log_probs - new_log_probs) ** 2).mean().item()
                
                # Compute gradients and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record metrics
                epoch_value_losses.append(value_loss.item())
                epoch_policy_losses.append(policy_loss.item())
                epoch_entropy_terms.append(entropy_term.item())
                epoch_kl_divs.append(approx_kl)
                
                batch_count += 1
                
                # Stop updating if KL divergence exceeds threshold
                if approx_kl > self.target_kl:
                    self.logger.info(f"Early stopping update as KL ({approx_kl:.6f}) exceeds target ({self.target_kl})")
                    break
            
            # Calculate average epoch metrics
            avg_value_loss = np.mean(epoch_value_losses)
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_entropy = np.mean(epoch_entropy_terms)
            avg_kl = np.mean(epoch_kl_divs)
            
            # Log epoch results
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.logger.info(f"Update epoch {epoch+1}/{self.update_epochs} completed: "
                          f"policy_loss={avg_policy_loss:.6f}, value_loss={avg_value_loss:.6f}, "
                          f"entropy={avg_entropy:.6f}, KL={avg_kl:.6f}, time={epoch_duration:.3f}s")
            
            # Record metrics
            value_losses.append(avg_value_loss)
            policy_losses.append(avg_policy_loss)
            entropy_terms.append(avg_entropy)
            kl_divs.append(avg_kl)
        
        # Add update results to metrics
        self.metrics['value_losses'].append(np.mean(value_losses))
        self.metrics['policy_losses'].append(np.mean(policy_losses))
        self.metrics['entropy'].append(np.mean(entropy_terms))
        self.metrics['kl'].append(np.mean(kl_divs))
        
        update_info = {
            'value_loss': np.mean(value_losses),
            'policy_loss': np.mean(policy_losses),
            'entropy': np.mean(entropy_terms),
            'kl_divergence': np.mean(kl_divs)
        }
        
        return update_info
    
    def save_model(self, path):
        """
        Save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'action_counter': self.action_counter,
            'current_action': self.current_action,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_bounds': self.action_bounds
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the model
        """
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Restore metrics and other states
                if 'metrics' in checkpoint:
                    self.metrics = checkpoint['metrics']
                if 'action_counter' in checkpoint:
                    self.action_counter = checkpoint['action_counter']
                if 'current_action' in checkpoint:
                    self.current_action = checkpoint['current_action']
                
                # Check dimensions and parameter ranges
                if 'state_dim' in checkpoint and checkpoint['state_dim'] != self.state_dim:
                    self.logger.warning(f"State space dimension mismatch: loaded={checkpoint['state_dim']}, current={self.state_dim}")
                if 'action_dim' in checkpoint and checkpoint['action_dim'] != self.action_dim:
                    self.logger.warning(f"Action space dimension mismatch: loaded={checkpoint['action_dim']}, current={self.action_dim}")
                
                self.logger.info(f"Model loaded from {path}")
            except Exception as e:
                self.logger.error(f"Model loading error: {e}")
                raise
        else:
            self.logger.error(f"Model file {path} not found")
            raise FileNotFoundError(f"Model file {path} not found")