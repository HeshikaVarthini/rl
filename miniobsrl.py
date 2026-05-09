import os
import time
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

class MinitaurRLEnv(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 3.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.obstacle_id = None
        self.step_counter = 0
        self.sensor_lines = []
        self.max_steps = 1000  # Max steps per episode
        super().__init__(render=render)
        
        # Extend observation space with sensor data
        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)
        
        # Action space remains the same (inherited from parent)

    def reset(self):
        self.step_counter = 0
        obs = super().reset()
        self._add_obstacle()
        return self._get_observation_with_sensors(obs)

    def _add_obstacle(self):
        if self.obstacle_id is not None:
            p.removeBody(self.obstacle_id)
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[1], base_matrix[2]])
        obstacle_position = np.array(base_position) + forward_vector * 2.5
        obstacle_position[2] = 0.25
        obstacle_urdf = pybullet_data.getDataPath() + "/cube.urdf"
        self.obstacle_id = p.loadURDF(
            obstacle_urdf,
            obstacle_position.tolist(),
            useFixedBase=True,
            globalScaling=0.5
        )
        p.changeVisualShape(self.obstacle_id, -1, rgbaColor=[1, 0, 0, 1])

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
            ray_dir = (np.cos(angle) * forward_vector + np.sin(angle) * right_vector)
            ray_dir /= np.linalg.norm(ray_dir)
            end = start + ray_dir * self.sensor_range
            result = p.rayTest(start, end)[0]
            hit_fraction = result[2] if result[0] == self.obstacle_id else 1.0
            sensor_data.append(hit_fraction)
            
            if self._is_render:
                color = [1, 0, 0] if result[0] == self.obstacle_id else [0, 1, 0]
                shortened_end = start + ray_dir * self.sensor_range * hit_fraction
                line_id = p.addUserDebugLine(start, shortened_end, color, lifeTime=0.1, lineWidth=3)
                self.sensor_lines.append(line_id)
                
        return np.concatenate([base_obs, sensor_data])

    def step(self, action):
        self.step_counter += 1
        base_obs, reward, done, info = super().step(action)
        
        # Get current position and orientation
        base_position, _ = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        x_position = base_position[0]
        
        # Reward for moving forward
        forward_reward = x_position * 0.1
        
        # Penalty for hitting the obstacle
        contact_points = p.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        collision_penalty = -100.0 if len(contact_points) > 0 else 0.0
        
        # Penalty for energy consumption (approximated by sum of squared actions)
        energy_penalty = -0.01 * np.sum(np.square(action))
        
        # Total reward
        reward = forward_reward + collision_penalty + energy_penalty
        
        # Check termination conditions
        done = done or self.step_counter >= self.max_steps or len(contact_points) > 0
        
        # Additional info
        info = {
            'x_position': x_position,
            'collision': len(contact_points) > 0,
            'steps': self.step_counter
        }
        
        return self._get_observation_with_sensors(base_obs), reward, done, info

def train_minitaur():
    # Create directory for models if it doesn't exist
    os.makedirs("./minitaur_models", exist_ok=True)
    
    # Create vectorized environment
    env = make_vec_env(lambda: MinitaurRLEnv(render=False), n_envs=4)
    
    # Define callbacks
    eval_callback = EvalCallback(
        MinitaurRLEnv(render=False),
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1),
        verbose=1,
        eval_freq=1000,
        best_model_save_path='./minitaur_models/'
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./minitaur_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0
    )
    
    # Train the model
    print("Starting training...")
    try:
        model.learn(total_timesteps=1_000_000, callback=eval_callback)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Save the final model
        model_path = "./minitaur_models/minitaur_ppo_final"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        env.close()

def test_minitaur(model_path="./minitaur_models/minitaur_ppo_final"):
    if not os.path.exists(model_path + ".zip"):
        print(f"Model file {model_path}.zip not found. Please train the model first.")
        return
    
    try:
        # Load the trained model
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path)
        
        # Create environment
        env = MinitaurRLEnv(render=True)
        
        # Run the trained agent
        obs = env.reset()
        done = False
        total_reward = 0
        episode_count = 0
        max_episodes = 5
        
        while episode_count < max_episodes:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, X Position: {info['x_position']:.2f}, Reward: {reward:.2f}")
            time.sleep(1.0/60.0)
            
            if done:
                print(f"Episode {episode_count + 1} completed! Total reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0
                episode_count += 1
        
        env.close()
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Minitaur RL Training/Testing')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode to run: "train" or "test" (default: train)')
    parser.add_argument('--model-path', type=str, default="./minitaur_models/minitaur_ppo_final",
                       help='Path to model for testing (default: ./minitaur_models/minitaur_ppo_final)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_minitaur()
    else:
        test_minitaur(args.model_path)