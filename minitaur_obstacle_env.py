import pybullet_envs.bullet.minitaur_gym_env as e
import pybullet as p
import numpy as np
import random
import gym

class MinitaurObstacleEnv(e.MinitaurBulletEnv):
    """
    Extended Minitaur environment with obstacles for avoidance training
    """
    def __init__(self, render=True, num_obstacles=10):
        # Initialize our attributes first
        self.num_obstacles = num_obstacles
        self.sensor_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        self.max_sensor_dist = 1.5
        self._first_reset = True  # Flag to handle first reset
        
        # Initialize obstacles list to prevent AttributeError
        self.obstacles = []
        
        # Temporarily replace reset method to handle initialization
        self._original_reset = self.reset
        self.reset = self._init_reset
        
        # Now call parent __init__
        super(MinitaurObstacleEnv, self).__init__(render=render)
        
        # Restore original reset method
        self.reset = self._original_reset
        
        # Override observation space to include sensor readings
        original_obs_dim = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(original_obs_dim + 8,),  # Add 8 sensor readings
            dtype=np.float32
        )
        
        # Now spawn obstacles after everything is initialized
        self.spawn_obstacles()
        
    def _init_reset(self):
        """Temporary reset method used during initialization"""
        # Call the parent's reset method directly without our custom logic
        # We'll use a flag to skip obstacle spawning during this first reset
        self._skip_obstacles = True
        state = e.MinitaurBulletEnv.reset(self)
        self._skip_obstacles = False
        return state
        
    def spawn_obstacles(self):
        """Spawn random obstacles in the environment"""
        # Clear existing obstacles
        for obstacle in self.obstacles:
            try:
                p.removeBody(obstacle)
            except:
                pass  # Ignore if body was already removed
        
        self.obstacles = []
        
        # Create new obstacles
        for _ in range(self.num_obstacles):
            x = random.uniform(1.0, 5.0)
            y = random.uniform(-2.0, 2.0)
            height = random.uniform(0.2, 0.5)
            width = random.uniform(0.3, 0.6)
            depth = random.uniform(0.3, 0.6)
            
            # Create obstacle using a simple shape
            collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, height/2])
            visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, height/2], rgbaColor=[0.7, 0.3, 0.3, 1])
            
            obstacle = p.createMultiBody(
                baseMass=0,  # Static obstacle
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=[x, y, height/2]
            )
            
            self.obstacles.append(obstacle)
    
    def get_distance_readings(self):
        """Get distance readings from 8 directional sensors"""
        pos, orn = p.getBasePositionAndOrientation(self.minitaur)
        readings = []
        
        for angle in self.sensor_angles:
            # Calculate ray direction in robot's local frame
            direction = [np.cos(angle), np.sin(angle), 0]
            
            # Rotate direction to world frame based on robot orientation
            rot_matrix = p.getMatrixFromQuaternion(orn)
            world_direction = [
                direction[0] * rot_matrix[0] + direction[1] * rot_matrix[1] + direction[2] * rot_matrix[2],
                direction[0] * rot_matrix[3] + direction[1] * rot_matrix[4] + direction[2] * rot_matrix[5],
                direction[0] * rot_matrix[6] + direction[1] * rot_matrix[7] + direction[2] * rot_matrix[8]
            ]
            
            # Cast ray and get distance to nearest obstacle
            result = p.rayTest(pos, 
                             [pos[0] + world_direction[0]*self.max_sensor_dist,
                              pos[1] + world_direction[1]*self.max_sensor_dist,
                              pos[2]])
            
            # Get distance (normalized between 0 and 1)
            distance = result[2] * self.max_sensor_dist
            readings.append(min(distance / self.max_sensor_dist, 1.0))
            
        return np.array(readings)
    
    def reset(self):
        """Reset the environment"""
        # Call parent reset
        state = super(MinitaurObstacleEnv, self).reset()
        
        # Skip obstacle spawning during initialization
        if not hasattr(self, '_skip_obstacles') or not self._skip_obstacles:
            # Clear and respawn obstacles
            self.spawn_obstacles()
            
            # Get sensor readings and add to observation
            distance_readings = self.get_distance_readings()
            extended_state = np.concatenate([state, distance_readings])
            return extended_state
        
        return state
    
    def step(self, action):
        """Execute action in the environment"""
        # Call parent step
        state, reward, done, info = super(MinitaurObstacleEnv, self).step(action)
        
        # Get sensor readings and add to observation
        distance_readings = self.get_distance_readings()
        extended_state = np.concatenate([state, distance_readings])
        
        # Modify reward to include obstacle avoidance
        min_distance = np.min(distance_readings)
        if min_distance < 0.3:  # If too close to obstacle
            reward -= 2.0 * (0.3 - min_distance)
        
        return extended_state, reward, done, info