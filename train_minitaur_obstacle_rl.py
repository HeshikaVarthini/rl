import os
import time
import numpy as np

# Set matplotlib backend before importing pyplot (for command prompt compatibility)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for command prompt
import matplotlib.pyplot as plt
from collections import deque

import gym
from gym import spaces

import pybullet as p
import pybullet_data

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
except Exception as import_error:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from import_error


# Import your environment class
from minisenchum1 import MinitaurWithSensors


class RewardLossCallback(BaseCallback):
    """
    Custom callback to track rewards and losses during training.
    """
    def __init__(self, save_dir: str, plot_freq: int = 1000, verbose: int = 0):
        super(RewardLossCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.plot_freq = plot_freq
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.timesteps = []
        
        # Current episode tracking (per environment)
        self.current_episode_rewards = None
        self.current_episode_lengths = None
        
        # Create plots directory
        self.plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Counter for plotting
        self.last_plot_step = 0
        
    def _on_training_start(self) -> None:
        """Initialize episode tracking arrays based on number of environments."""
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = np.zeros(n_envs)
        self.current_episode_lengths = np.zeros(n_envs, dtype=int)
        
    def _on_step(self) -> bool:
        # Get rewards and dones from the step
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        
        # Update current episode stats
        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1
        
        # Check for completed episodes
        for i, done in enumerate(dones):
            if done:
                # Episode finished, record it
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lengths.append(self.current_episode_lengths[i])
                self.timesteps.append(self.num_timesteps)
                
                if self.verbose > 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"Reward={self.current_episode_rewards[i]:.2f}, "
                          f"Length={self.current_episode_lengths[i]}, "
                          f"Avg(last 10)={avg_reward:.2f}")
                
                # Reset for next episode
                self.current_episode_rewards[i] = 0
                self.current_episode_lengths[i] = 0
        
        # Plot and save graphs periodically
        if self.num_timesteps - self.last_plot_step >= self.plot_freq:
            if len(self.episode_rewards) > 0:
                print(f"\n{'='*60}")
                print(f"Generating plot at timestep {self.num_timesteps}...")
                print(f"Episodes completed so far: {len(self.episode_rewards)}")
                print(f"{'='*60}")
                self._plot_and_save()
                self.last_plot_step = self.num_timesteps
            else:
                print(f"\n⚠ Timestep {self.num_timesteps}: No episodes completed yet, skipping plot...")
            
        return True
    
    def _plot_and_save(self):
        """Plot and save reward and loss graphs."""
        if len(self.episode_rewards) < 1:
            if self.verbose > 0:
                print(f"Not enough episodes yet ({len(self.episode_rewards)}), skipping plot...")
            return
            
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Plot 1: Episode Rewards
        ax1.plot(self.episode_rewards, 'b-', alpha=0.7, linewidth=1)
        if len(self.episode_rewards) > 10:
            # Moving average
            window = min(50, len(self.episode_rewards) // 4)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            ax1.legend()
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2.plot(self.episode_lengths, 'g-', alpha=0.7, linewidth=1)
        if len(self.episode_lengths) > 10:
            window = min(50, len(self.episode_lengths) // 4)
            moving_avg = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            ax2.legend()
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reward Distribution (histogram)
        ax3.hist(self.episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(self.episode_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax3.axvline(np.median(self.episode_rewards), color='green', linestyle='--', 
                   label=f'Median: {np.median(self.episode_rewards):.2f}')
        ax3.set_title('Reward Distribution')
        ax3.set_xlabel('Total Reward')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Statistics
        ax4.text(0.1, 0.8, f'Total Episodes: {len(self.episode_rewards)}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Total Timesteps: {self.num_timesteps:,}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Mean Reward: {np.mean(self.episode_rewards):.2f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'Std Reward: {np.std(self.episode_rewards):.2f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f'Max Reward: {np.max(self.episode_rewards):.2f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.3, f'Min Reward: {np.min(self.episode_rewards):.2f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, f'Mean Length: {np.mean(self.episode_lengths):.1f}', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Training Statistics')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, f'training_progress_{self.num_timesteps}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose > 0:
            print(f"Training plots saved to: {plot_path}")
    
    def _on_training_end(self):
        """Save final plots when training ends."""
        print(f"\n{'='*60}")
        print(f"Training ended. Generating final plots...")
        print(f"Total episodes completed: {len(self.episode_rewards)}")
        print(f"{'='*60}")
        
        if len(self.episode_rewards) == 0:
            print("⚠ WARNING: No episodes completed during training!")
            print("This means episodes are too long or never reached done=True")
            print("Cannot generate plots without completed episodes.")
            return
        
        # Always generate final plot
        self._plot_and_save()
        
        # Save data to CSV for further analysis
        try:
            import pandas as pd
            data = {
                'episode': range(1, len(self.episode_rewards) + 1),
                'reward': self.episode_rewards,
                'length': self.episode_lengths,
                'timestep': self.timesteps[:len(self.episode_rewards)]
            }
            df = pd.DataFrame(data)
            csv_path = os.path.join(self.plots_dir, 'training_data.csv')
            df.to_csv(csv_path, index=False)
            
            print(f"✓ Training data saved to: {csv_path}")
            print(f"✓ Total episodes recorded: {len(self.episode_rewards)}")
            print(f"✓ Average reward: {np.mean(self.episode_rewards):.2f}")
            print(f"✓ Best reward: {np.max(self.episode_rewards):.2f}")
        except ImportError:
            # If pandas not available, save as numpy arrays
            csv_path = os.path.join(self.plots_dir, 'training_data.npz')
            np.savez(csv_path, 
                    episodes=np.arange(1, len(self.episode_rewards) + 1),
                    rewards=np.array(self.episode_rewards),
                    lengths=np.array(self.episode_lengths),
                    timesteps=np.array(self.timesteps[:len(self.episode_rewards)]))
            print(f"✓ Training data saved to: {csv_path} (numpy format)")
            print(f"✓ Total episodes recorded: {len(self.episode_rewards)}")


class ActionBiasScaleWrapper(gym.Wrapper):
    """
    Stabilize gait by scaling actions and adding knee flexion bias.
    Keeps final actions within the original Box limits.
    """
    def __init__(self, env: gym.Env, knee_bias: float = 0.3, scale: float = 0.8):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        self.knee_bias = float(knee_bias)
        self.scale = float(scale)
        self._last_action = np.zeros(env.action_space.shape, dtype=np.float32)

        # Precompute bias vector for knee joints
        num_joints = env.action_space.shape[0]
        bias = np.zeros(num_joints, dtype=np.float32)
        knee_indices = [1, 3, 5, 7] if num_joints >= 8 else list(range(1, num_joints, 2))
        for idx in knee_indices:
            bias[idx] = self.knee_bias
        self._bias = bias

    def reset(self, **kwargs):
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.tanh(action)  # ensure within [-1, 1]
        action = self.scale * action + self._bias
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self._last_action = action
        return self.env.step(action)


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds reward for forward progress and obstacle avoidance, plus posture and smoothness terms
    to encourage a stable trot with lifted knees.

    Observation is unchanged (base obs + 8 sensor fractions). Reward becomes:
        reward = forward_gain * delta_x
                 + obstacle_avoidance_bonus
                 + stability_bonus
                 - collision_penalty
                 + alive_bonus
    where obstacle_avoidance_bonus rewards staying away from obstacles.
    """

    def __init__(self, env: gym.Env,
                 forward_gain: float = 3.0,
                 obstacle_avoidance_bonus: float = 2.0,
                 collision_cost: float = 50.0,
                 alive_bonus: float = 0.1,
                 obstacle_threshold_m: float = 1.0,
                 stability_bonus: float = 0.5,
                 smooth_weight: float = 0.02):
        super().__init__(env)
        self.forward_gain = forward_gain
        self.obstacle_avoidance_bonus = obstacle_avoidance_bonus
        self.collision_cost = collision_cost
        self.alive_bonus = alive_bonus
        self.obstacle_threshold_m = obstacle_threshold_m
        self.stability_bonus = stability_bonus
        self.smooth_weight = smooth_weight

        self._last_base_x: float = 0.0
        self._last_base_pos = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Expect last 8 obs entries to be sensor fractions in [0,1]
        assert isinstance(self.observation_space, spaces.Box)
        assert self.observation_space.shape[0] >= 8, "Observation must include 8 sensor readings."

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_base_x = self._get_base_x()
        base_pos, _ = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        self._last_base_pos = np.array(base_pos, dtype=np.float32)
        self._last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)

        # Forward progress aligned with robot heading
        base_pos, base_ori = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        disp = np.array(base_pos, dtype=np.float32) - self._last_base_pos
        self._last_base_pos = np.array(base_pos, dtype=np.float32)

        base_mat = p.getMatrixFromQuaternion(base_ori)
        body_forward = np.array([base_mat[0], base_mat[3], base_mat[6]], dtype=np.float32)
        norm = np.linalg.norm(body_forward) + 1e-8
        body_forward /= norm
        forward_progress = float(np.dot(disp, body_forward))

        # Keep delta_x for debugging
        current_x = float(base_pos[0])
        delta_x = current_x - self._last_base_x
        self._last_base_x = current_x

        forward_reward = self.forward_gain * forward_progress

        # Obstacle avoidance bonus using front/left/right sensors (dist fractions, convert to meters)
        sensor_fractions = obs[-8:]
        # Angles: [-pi, -3pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3pi/4]
        front_fraction = sensor_fractions[4]
        left_fraction = sensor_fractions[2]
        right_fraction = sensor_fractions[6]

        # Convert fraction to distance in meters using env.sensor_range
        sensor_range_m = getattr(self.env, 'sensor_range', 2.0)
        front_m = float(front_fraction) * sensor_range_m
        left_m = float(left_fraction) * sensor_range_m
        right_m = float(right_fraction) * sensor_range_m

        min_front_lr = min(front_m, left_m, right_m)
        # Reward for staying away from obstacles (positive reward)
        obstacle_avoidance_reward = self.obstacle_avoidance_bonus * min(1.0, min_front_lr / self.obstacle_threshold_m)

        # Collision penalty if robot touches obstacle
        collision_happened = self._has_collision_with_obstacle()
        collision_penalty = self.collision_cost if collision_happened else 0.0

        # Stability bonus (reward for good posture - low roll/pitch)
        _, base_ori = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        roll, pitch, _ = p.getEulerFromQuaternion(base_ori)
        posture_angle = abs(roll) + abs(pitch)
        stability_reward = self.stability_bonus * max(0.0, 1.0 - posture_angle / (np.pi/4))  # Max reward when posture is good

        # Smoothness penalty (discourage large action deltas) - keep this as penalty
        if isinstance(action, np.ndarray):
            action_np = action.astype(np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)
        smooth_penalty = self.smooth_weight * float(np.linalg.norm(action_np - self._last_action, ord=1))
        self._last_action = action_np

        shaped_reward = (
            forward_reward
            + obstacle_avoidance_reward
            + stability_reward
            - collision_penalty
            - smooth_penalty
            + self.alive_bonus
        )

        # Optionally terminate on collision
        if collision_happened:
            done = True

        # Expose diagnostics
        info = dict(info)
        info.update({
            'delta_x': delta_x,
            'forward_progress': forward_progress,
            'forward_reward': forward_reward,
            'obstacle_avoidance_reward': obstacle_avoidance_reward,
            'stability_reward': stability_reward,
            'collision_penalty': float(collision_penalty),
            'front_m': front_m,
            'left_m': left_m,
            'right_m': right_m,
            'posture_angle': posture_angle,
            'smooth_penalty': smooth_penalty,
        })

        return obs, shaped_reward, done, info

    def _get_base_x(self) -> float:
        # Compatible with underlying MinitaurWithSensors that uses pybullet directly
        base_pos, _ = p.getBasePositionAndOrientation(self.env.minitaur.quadruped)
        return float(base_pos[0])

    def _has_collision_with_obstacle(self) -> bool:
        try:
            contact_points = p.getContactPoints(self.env.minitaur.quadruped, self.env.obstacle_id)
            return len(contact_points) > 0
        except Exception:
            return False


def make_env(render: bool = False) -> gym.Env:
    base_env = MinitaurWithSensors(render=render)
    # Apply action bias/scale first, then reward shaping
    gait_env = ActionBiasScaleWrapper(base_env, knee_bias=0.3, scale=0.8)
    wrapped_env = RewardShapingWrapper(gait_env)
    return wrapped_env


def create_live_plot():
    """Create a live plotting window for real-time training monitoring."""
    # Note: Live plotting disabled when using 'Agg' backend (command prompt mode)
    # Plots will be saved to disk instead
    return None


def main():
    num_envs = int(os.environ.get('NUM_ENVS', '1'))
    total_timesteps = int(os.environ.get('TOTAL_STEPS', str(400_000)))
    save_dir = os.environ.get('SAVE_DIR', os.path.join('checkpoints', 'ppo_minitaur_obstacle'))
    os.makedirs(save_dir, exist_ok=True)
    
    # Note: Live plotting disabled for command prompt compatibility
    # All plots will be saved to disk instead
    print("Plotting mode: Saving plots to disk (command prompt compatible)")
    print(f"Plots will be saved to: {os.path.join(save_dir, 'plots')}")
    live_plot = None

    if num_envs > 1:
        def _make():
            return make_env(render=False)
        vec_env = SubprocVecEnv([_make for _ in range(num_envs)])
    else:
        vec_env = DummyVecEnv([lambda: make_env(render=False)])

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000 // num_envs,
        save_path=save_dir,
        name_prefix='ppo_minitaur_obs'
    )
    
    # Add reward and loss tracking callback
    reward_loss_callback = RewardLossCallback(
        save_dir=save_dir,
        plot_freq=2_000,  # Plot every 2000 timesteps (more frequent for testing)
        verbose=1
    )
    
    print(f"Callback initialized. Plots will be saved every 2,000 timesteps.")
    print(f"Plot directory: {os.path.join(save_dir, 'plots')}")
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([checkpoint_callback, reward_loss_callback])

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // max(1, num_envs),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=os.path.join(save_dir, 'tb')
    )

    print("Starting training...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {num_envs}")
    print(f"Save directory: {save_dir}")
    print("=" * 50)
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = os.path.join(save_dir, 'ppo_minitaur_obs_final')
    model.save(final_path)
    print(f"Saved final model to: {final_path}")
    
    # Close live plot if it was created
    if live_plot is not None:
        try:
            plt.close(live_plot[0])
        except:
            pass
    
    print("=" * 50)
    print("Training completed!")
    print(f"Check the following directories for results:")
    print(f"  - Model checkpoints: {save_dir}")
    print(f"  - Training plots: {os.path.join(save_dir, 'plots')}")
    print(f"  - Tensorboard logs: {os.path.join(save_dir, 'tb')}")
    print("=" * 50)


if __name__ == '__main__':
    main()
