import gymnasium as gym
import pybullet as p
import pybullet_data
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

class MinitaurObstacleAvoidanceEnv(MinitaurBulletEnv):
    """
    A custom Gymnasium environment for the Minitaur robot with obstacles.

    This class extends MinitaurBulletEnv by adding:
    1.  Cubic obstacles to the environment.
    2.  A simulated LIDAR sensor to the robot's observation space.
    3.  A custom reward function that penalizes collisions.
    """
    def __init__(self, **kwargs):
        # Call the parent class's constructor
        super(MinitaurObstacleAvoidanceEnv, self).__init__(**kwargs)
        
        # Define obstacle properties
        self.num_obstacles = 15
        self.obstacle_positions = []
        
        # Define LIDAR properties
        self.num_rays = 12  # Number of simulated LIDAR rays
        self.ray_length = 5.0  # Max distance to check for obstacles
        
        # The new observation space is a combination of the original state and LIDAR data
        base_obs_space_shape = self.observation_space.shape[0]
        new_obs_space_shape = base_obs_space_shape + self.num_rays
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_space_shape,), dtype=np.float32
        )

    def _setup_pybullet(self):
        """Set up the PyBullet simulation and add obstacles."""
        super()._setup_pybullet()
        self._create_obstacles()

    def _create_obstacles(self):
        """Randomly place cube obstacles in the environment."""
        
        # Create a single collision shape for all obstacles to save resources
        box_half_extents = [0.2, 0.2, 0.5]
        box_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        
        # Place obstacles around the starting point of the robot
        for i in range(self.num_obstacles):
            # Randomize position within a certain radius
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            z = box_half_extents[2]  # Obstacle should be on the ground
            position = [x, y, z]
            
            # Create the obstacle instance in the simulation
            box_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=box_collision_shape,
                baseVisualShapeIndex=-1, # No visual shape
                basePosition=position,
            )
            self.obstacle_positions.append(box_id)

    def _get_lidar_data(self):
        """Simulate a LIDAR sensor using PyBullet's ray casting."""
        
        base_pos, _ = p.getBasePositionAndOrientation(self.robot.quadruped)
        
        lidar_readings = []
        for i in range(self.num_rays):
            angle = (2 * np.pi / self.num_rays) * i
            ray_to = [
                base_pos[0] + self.ray_length * np.cos(angle),
                base_pos[1] + self.ray_length * np.sin(angle),
                base_pos[2] # Keep the ray at the same height
            ]
            
            # Cast a single ray and get the hit information
            result = p.rayTest(base_pos, ray_to)
            hit_fraction = result[0][2]
            
            # Convert hit fraction to a distance
            distance = self.ray_length * hit_fraction
            lidar_readings.append(distance)
            
        return np.array(lidar_readings, dtype=np.float32)

    def _get_obs(self):
        """Get the combined observation: original Minitaur state + LIDAR data."""
        base_obs = super()._get_obs()
        lidar_data = self._get_lidar_data()
        
        # Concatenate the observations
        combined_obs = np.concatenate([base_obs, lidar_data])
        return combined_obs

    def _reward(self):
        """Calculate the custom reward based on forward progress and collisions."""
        
        # Original reward from the Minitaur environment (e.g., forward progress)
        base_reward = super()._reward()

        # Get robot position to check for collisions
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot.quadruped)
        
        # Check for collision with any obstacle
        for obstacle_id in self.obstacle_positions:
            contact_points = p.getContactPoints(self.robot.quadruped, obstacle_id)
            if contact_points:
                # Assign a large negative reward for collision
                print("Collision! 💥")
                return -500.0
                
        # Return the original reward if no collision occurred
        return base_reward

    def _reset(self):
        """Reset the simulation and re-place the obstacles."""
        obs = super()._reset()
        # Remove old obstacles before resetting to a new position
        for ob_id in self.obstacle_positions:
            p.removeBody(ob_id)
        self.obstacle_positions = []
        self._create_obstacles()
        
        # Return the new combined observation
        return self._get_obs()

# --- Training and Demonstration ---

if __name__ == "__main__":
    
    print("Initializing environment...")
    
    # Create the custom environment
    env = MinitaurObstacleAvoidanceEnv(render=True, is_alive_at_zero=False)
    
    # Wrap the environment in a vectorized environment for Stable Baselines3
    # DummyVecEnv is used for single-threaded training
    env = DummyVecEnv([lambda: env])

    # Instantiate the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, learning_rate=3e-4)

    print("Starting training...")
    # Train the agent for a specified number of timesteps
    model.learn(total_timesteps=100_000)

    print("Training finished. Demonstrating trained agent...")
    
    # Save the trained model
    model.save("minitaur_obstacle_avoidance")
    
    # Demonstrate the trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()

    env.close()