import pybullet as p
import numpy as np
import gym
from gym import spaces
from env.quad_env import RobotEnvironment
from env.gait import GaitGenerator
from env.config_loader import ConfigLoader
import logging
import time

class QuadrupedEnv(gym.Env):
    """
    Reinforcement learning environment for quadruped robots
    
    Wraps the simulation environment as a Gym environment,
    providing an environment for learning obstacle avoidance and goal reaching tasks.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_file=None, render=True, max_steps=1000, logger=None):
        """
        Initialize the environment
        
        Args:
            config_file (str, optional): Path to configuration file. Default settings used if None.
            render (bool): Whether to render
            max_steps (int): Maximum number of steps per episode
            logger (logging.Logger): Logger instance
        """
        super(QuadrupedEnv, self).__init__()
        
        # Logger setup
        self.logger = logger or logging.getLogger('rl_environment')
        
        # Load configuration
        self.config = ConfigLoader(config_file)
        
        # Rendering settings
        self.render_enabled = render
        
        # Update environment settings - override rendering settings
        env_config = self.config.get('environment')
        env_config['use_gui'] = render
        env_config['camera_follow'] = render
        
        # Episode limit
        self.max_steps = max_steps
        self.current_step = 0
        
        # Robot environment and simulation instances
        self.env = None
        self.gait_generator = None
        self.setup_environment()
        
        # Action space definition (gait parameters)
        # [gait amplitude, gait frequency, turn direction, turn intensity]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.8, -1.0, 0.0]),
            high=np.array([0.5, 2.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space definition
        # [position(x,y,z), orientation(roll,pitch,yaw), 
        #  goal direction(x,y), distance to goal, 
        #  linear velocity(x,y,z),
        #  obstacle sensors (8 directions)]
        lidar_sectors = 8  # 8-direction obstacle sensors
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.inf, -np.inf, -np.inf] +  # position
                [-np.pi, -np.pi, -np.pi] +     # orientation
                [-1.0, -1.0] +                # goal direction
                [0.0] +                       # distance to goal
                [-5.0, -5.0, -5.0] +          # linear velocity (x,y,z)
                [0.0] * lidar_sectors         # LIDAR sectors
            ),
            high=np.array(
                [np.inf, np.inf, np.inf] +     # position
                [np.pi, np.pi, np.pi] +        # orientation
                [1.0, 1.0] +                  # goal direction
                [np.inf] +                    # distance to goal
                [5.0, 5.0, 5.0] +             # linear velocity (x,y,z)
                [np.inf] * lidar_sectors      # LIDAR sectors
            ),
            dtype=np.float32
        )
        
        # Record state dimension
        self.state_dim = self.observation_space.shape[0]
        
        # Reward function weights
        self.reward_weights = {
            'goal_distance': 1000000,   # Weight for reward of getting closer to goal
            'obstacle_penalty': 0.005,  # Weight for penalty of getting close to obstacles
            'energy_penalty': 0.0005,   # Weight for penalty of energy consumption
            'goal_reward': 5.0,         # Reward for reaching goal
            'goal_time_bonus': 0.05,    # Time bonus coefficient for goal achievement
            'fail_penalty': 8.0,        # Penalty for not reaching the goal
            'stability_reward': 0.0005, # Stability reward
            'time_penalty': 0.001,      # Time progression penalty
            'velocity_reward': 0.01,    # Velocity reward
            'safe_distance_bonus': 6.5, # Safe distance bonus
            'direction_alignment': 0.01 # Reward for alignment toward goal direction
        }
                        
        # Store last observation and action
        self.last_observation = None
        self.last_action = None
        self.last_goal_distance = None
        
        # Statistics
        self.episode_steps = 0
        self.total_reward = 0.0
        
        # Variables for time tracking
        self.start_time = None
        self.goal_time = None
        
        self.logger.info(f"QuadrupedEnv initialization complete: render={render}, max_steps={max_steps}, config={config_file}")
        self.logger.info(f"Action space: {self.action_space}")
        self.logger.info(f"Observation space: {self.observation_space}")
    
    def setup_environment(self):
        """Set up the environment"""
        # Get robot environment settings
        robot_config = self.config.get('robot')
        env_config = self.config.get('environment')
        
        # Initialize environment
        urdf_path = robot_config['urdf_path']
        robot_pos = robot_config['position']
        robot_rot = robot_config['rotation']
        
        self.logger.info(f"Initializing environment: URDF={urdf_path}, pos={robot_pos}, rot={robot_rot}")
        
        # Initialize environment (rendering depends on render_enabled)
        self.env = RobotEnvironment(
            urdf_path=urdf_path, 
            robot_pos=robot_pos, 
            robot_rot=robot_rot, 
            use_gui=self.render_enabled, 
            cam=self.render_enabled
        )
        
        # Set physics parameters for the environment
        p.setGravity(*env_config['gravity'])
        p.setTimeStep(env_config['timestep'])
        
        # Initial gait generator setup
        gait_config = self.config.get('gait')
        self.gait_generator = GaitGenerator(
            amplitude=gait_config.get('amplitude', 0.25),  # Default amplitude
            frequency=gait_config.get('frequency', 1.5)   # Default frequency
        )
        self.gait_generator.set_turn_direction(gait_config['turn_direction'])
        self.gait_generator.set_turn_intensity(gait_config['turn_intensity'])
        self.gait_generator.set_backward(False)  # Forward movement by default
        self.gait_generator.set_gait_pattern(gait_config['pattern'])
        
        # LiDAR sensor setup
        lidar_config = self.config.get('lidar')
        if lidar_config['enabled']:
            self.lidar = self.env.add_lidar(
                num_rays=lidar_config['num_rays'],
                ray_length=lidar_config['ray_length'],
                ray_start_length=lidar_config['ray_start_length']
            )
            self.logger.info(f"LiDAR sensor initialized: rays={lidar_config['num_rays']}, length={lidar_config['ray_length']}")
        
        # Obstacle setup
        obstacle_config = self.config.get('obstacles')
        if obstacle_config['enabled']:
            self.obstacles = self.env.add_obstacles(
                course_type=obstacle_config['course_type'],
                length=obstacle_config['length']
            )
            self.logger.info(f"Obstacle course initialized: type={obstacle_config['course_type']}, length={obstacle_config['length']}")
        
        # Goal setup
        goal_config = self.config.get('goal')
        if goal_config['enabled']:
            self.goal = self.env.add_goal(
                goal_position=goal_config['position'],
                radius=goal_config['radius'],
                color=goal_config['color'] if 'color' in goal_config else None
            )
            self.logger.info(f"Goal initialized: position={goal_config['position']}, radius={goal_config['radius']}")
    
    def reset(self):
        """
        Reset the environment to initial state
        
        Returns:
            numpy.ndarray: Initial observation
        """
        # Clean up existing environment
        if self.env:
            self.env.close()
        
        # Re-initialize environment
        self.setup_environment()
        
        # Reset step counter
        self.current_step = 0
        self.episode_steps = 0
        self.total_reward = 0.0
        
        # Start time measurement
        self.start_time = time.time()
        self.goal_time = None
        
        self.logger.info("Environment has been reset")
        
        # Get initial observation
        observation = self._get_observation()
        self.last_observation = observation
        
        # Record initial goal distance
        state = self.env.get_full_state()
        if 'goal' in state:
            self.last_goal_distance = state['goal']['distance']
            self.logger.info(f"Initial goal distance: {self.last_goal_distance}")
        
        return observation
    
    def step(self, action):
        """
        Execute an action in the environment
        
        Args:
            action (numpy.ndarray): Action to execute [amplitude, frequency, turn direction, turn intensity]
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.last_action = action.copy()
        
        # Convert action to gait parameters
        amplitude, frequency, turn_direction, turn_intensity = action
        
        # Log action details
        self.logger.debug(f"Action: amplitude={amplitude:.3f}, frequency={frequency:.3f}, direction={turn_direction:.3f}, intensity={turn_intensity:.3f}")
        
        # Set gait generator parameters
        self.gait_generator.amplitude = amplitude
        self.gait_generator.frequency = frequency
        self.gait_generator.set_turn_direction(turn_direction)
        self.gait_generator.set_turn_intensity(turn_intensity)
        
        # Generate gait based on parameters and execute multiple steps
        # Execute multiple simulation steps per action for gait stability
        steps_per_action = 20  # Number of steps equivalent to one gait cycle
        max_force = self.config.get('robot')['max_force']
        
        # Execute simulation steps
        done = False
        info = {}
        
        for _ in range(steps_per_action):
            # Get action from gait generator
            gait_action = self.gait_generator.get_action()
            
            # Apply action to robot
            self.env.apply_action(gait_action, max_force)
            
            # Advance simulation by one step
            self.env.step()
            
            # Check if goal reached
            if self.env.check_goal_reached():
                done = True
                info['goal_reached'] = True
                self.goal_time = time.time() - self.start_time
                info['goal_time'] = self.goal_time
                self.logger.info(f"Goal reached! Time taken: {self.goal_time:.2f} seconds")
                break
            
            # Check for fall over
            robot_state = self.env.get_robot_state()
            _, orientation = robot_state['base_position'], robot_state['base_orientation']
            
            # Fall detection (when robot is significantly tilted)
            import pybullet as p
            euler = p.getEulerFromQuaternion(orientation)
            roll, pitch = euler[0], euler[1]
            
            # Consider robot fallen if significantly tilted
            if abs(roll) > 0.8 or abs(pitch) > 0.8:
                done = True
                info['fell_over'] = True
                self.logger.info(f"Robot fell over: roll={roll:.3f}, pitch={pitch:.3f}")
                break
        
        # Update step counter
        self.current_step += 1
        self.episode_steps += 1
        
        # Check if maximum steps reached
        if self.current_step >= self.max_steps:
            done = True
            info['timeout'] = True
            self.logger.info(f"Reached maximum steps ({self.max_steps})")
        
        # Get next observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(observation, done, info)
        self.total_reward += reward
        
        # Add step count and cumulative reward to info
        info['steps'] = self.episode_steps
        info['total_reward'] = self.total_reward
        
        # Log periodically
        if self.episode_steps % 100 == 0 or done:
            robot_state = self.env.get_robot_state()
            self.logger.info(f"Step {self.episode_steps}: position={robot_state['base_position']}, reward={reward:.3f}, total_reward={self.total_reward:.3f}")
        
        # Update last observation
        self.last_observation = observation
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """
        Get observation from current state
        """
        # Get robot state
        state = self.env.get_full_state()
        
        # Basic position and orientation
        base_position = state['base_position']
        import pybullet as p
        euler = p.getEulerFromQuaternion(state['base_orientation'])
        
        observation = []
        
        # Position information [x, y, z]
        observation.extend(base_position)
        
        # Orientation information [roll, pitch, yaw]
        observation.extend(euler)
        
        # Relative position and distance to goal
        if 'goal' in state:
            goal_position = state['goal']['position']
            goal_distance = state['goal']['distance']
            
            # Direction vector to goal (normalized)
            goal_direction = [
                goal_position[0] - base_position[0],
                goal_position[1] - base_position[1]
            ]
            
            # Normalize direction vector
            norm = np.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
            if norm > 0:
                goal_direction = [goal_direction[0]/norm, goal_direction[1]/norm]
            
            # Add goal information
            observation.extend(goal_direction)  # Goal direction
            observation.append(goal_distance)   # Distance to goal
            
            # Update last goal distance
            self.last_goal_distance = goal_distance
        else:
            # Add dummy data if no goal information
            observation.extend([0.0, 0.0])  # Direction
            observation.append(10.0)        # Distance
        
        # Velocity information [vx, vy, vz]
        if 'base_linear_velocity' in state:
            observation.extend(state['base_linear_velocity'])
        else:
            observation.extend([0.0, 0.0, 0.0])  # Default values
        
        # LiDAR sector information (distance to obstacles)
        if 'lidar' in state and 'sectors' in state['lidar']:
            sectors = state['lidar']['sectors']
            # Add minimum distance for each sector to observation
            sector_distances = [sector['min_distance'] for sector in sectors]
            
            # Normalization: higher values for closer distances (range 0 to 1)
            # This makes neural network inputs larger for closer obstacles, increasing their influence
            normalized_distances = [min(1.0, 1.0 / (dist + 0.1)) for dist in sector_distances]
            observation.extend(normalized_distances)
        else:
            # Add dummy data if no LiDAR information (no obstacles = 0.0)
            observation.extend([0.0] * 8)  # No obstacles in all 8 directions
        
        return np.array(observation, dtype=np.float32)
    
    def _compute_reward(self, observation, done, info):
        """
        Calculate reward
        
        Args:
            observation (numpy.ndarray): Current observation
            done (bool): Episode termination flag
            info (dict): Additional information
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        reward_details = {}
        
        # Goal achievement reward (with time bonus)
        if 'goal_reached' in info and info['goal_reached']:
            # Base goal reward
            goal_reward = self.reward_weights['goal_reward']
            reward += goal_reward
            reward_details['goal_reward'] = goal_reward
            
            # Time bonus calculation: larger reward for faster completion
            if 'goal_time' in info:
                # Assume max_steps is the normally required time
                normalized_time = 1.0 - (info['goal_time'] / (self.max_steps * 0.05))  # 0.05 is equivalent to time step
                normalized_time = max(0.0, normalized_time)  # Ensure non-negative
                time_bonus = normalized_time * self.reward_weights['goal_time_bonus'] * self.reward_weights['goal_reward']
                reward += time_bonus
                reward_details['time_bonus'] = time_bonus
                self.logger.info(f"Time bonus: {time_bonus:.2f} (completion time: {info['goal_time']:.2f} seconds)")
        
        # Timeout penalty (when goal not reached)
        if done and not ('goal_reached' in info and info['goal_reached']):
            if 'timeout' in info and info['timeout']:
                fail_penalty = -self.reward_weights['fail_penalty']
                reward += fail_penalty
                reward_details['fail_penalty'] = fail_penalty
                self.logger.info(f"Goal not reached penalty: {fail_penalty:.2f}")
        
        # Fall over penalty
        if 'fell_over' in info and info['fell_over']:
            fall_penalty = -self.reward_weights['goal_reward'] / 2
            reward += fall_penalty
            reward_details['fall_penalty'] = fall_penalty
        
        # Reward based on distance to goal
        if self.last_goal_distance is not None:
            current_goal_distance = observation[8]  # Distance to goal is at index 8
            
            # Reward for getting closer to goal
            distance_improvement = self.last_goal_distance - current_goal_distance
            distance_reward = distance_improvement * self.reward_weights['goal_distance']
            reward += distance_reward
            reward_details['distance_reward'] = distance_reward
            
            # Penalty for moving away from goal
            if distance_improvement < 0:  # If moved away from goal
                distance_penalty = abs(distance_improvement) * self.reward_weights['goal_distance'] * 5.0  # 5x penalty for moving away
                reward -= distance_penalty
                reward_details['distance_penalty'] = -distance_penalty
            
            # Save current distance
            self.last_goal_distance = current_goal_distance
        
        # Reward for robot orientation aligned with goal direction
        yaw = observation[5]  # Yaw angle is at index 5
        goal_direction = observation[6:8]  # Goal direction vector is at indices 6-7
        
        # Robot forward direction vector (calculated from yaw angle)
        robot_forward = [np.cos(yaw), np.sin(yaw)]
        
        # Calculate dot product of robot forward direction and goal direction (cosine similarity)
        # Close to 1 means same direction, close to -1 means opposite direction
        direction_alignment = robot_forward[0] * goal_direction[0] + robot_forward[1] * goal_direction[1]
        
        # Reward based on direction alignment (convert range -1 to 1 into 0 to 1)
        direction_reward = (direction_alignment + 1) / 2 * self.reward_weights.get('direction_alignment', 0.5)
        reward += direction_reward
        reward_details['direction_reward'] = direction_reward
        
        # Penalty based on distance to obstacles (by direction)
        obstacle_distances = observation[12:]  # Obstacle distances start at index 12 after velocity information
        
        # Special penalty for front obstacles (front is index 0)
        front_obstacle_distance = obstacle_distances[0]  # Front sector
        side_obstacle_distances = [obstacle_distances[1], obstacle_distances[7]]  # Side sectors
        rear_obstacle_distance = obstacle_distances[4]  # Rear sector

        # Larger penalty when front obstacle is very close
        if front_obstacle_distance > 0.5:  # Judged by normalized value (larger value means closer obstacle)
            front_penalty = front_obstacle_distance * self.reward_weights['obstacle_penalty'] * 2.0
            reward -= front_penalty
            reward_details['front_obstacle_penalty'] = -front_penalty
        
        # Also penalize side obstacles (but less than front)
        side_obstacle_value = max(side_obstacle_distances)
        if side_obstacle_value > 0.3:  # Judged by normalized value
            side_penalty = side_obstacle_value * self.reward_weights['obstacle_penalty'] * 1.0
            reward -= side_penalty
            reward_details['side_obstacle_penalty'] = -side_penalty
        
        # Reward based on velocity (encourage faster movement)
        velocity = observation[9:12]  # Velocity information is at indices 9 to 11
        forward_velocity = velocity[0]  # X-direction velocity (forward direction)
        
        # Proportional reward for positive forward velocity
        if forward_velocity > 0:
            # Base velocity reward
            velocity_reward = forward_velocity * self.reward_weights.get('velocity_reward', 1.0)
            reward += velocity_reward  # Add to reward
            reward_details['velocity_reward'] = velocity_reward
        
        # Reward based on posture stability
        roll, pitch = observation[3], observation[4]  # Roll and pitch angles are at indices 3 and 4
        stability = 1.0 - (abs(roll) + abs(pitch)) / (2 * np.pi)  # Normalize to 0~1 range
        stability_reward = stability * self.reward_weights['stability_reward']
        reward += stability_reward
        reward_details['stability_reward'] = stability_reward
        
        # Energy penalty based on action magnitude
        if self.last_action is not None:
            # Higher amplitude and frequency mean more energy consumption
            energy_consumption = self.last_action[0] * self.last_action[1]  # amplitude × frequency
            energy_penalty = energy_consumption * self.reward_weights['energy_penalty']
            reward -= energy_penalty
            reward_details['energy_penalty'] = -energy_penalty
        
        # Time progression penalty (to avoid prolonged exploration)
        time_penalty = self.reward_weights['time_penalty']
        reward -= time_penalty
        reward_details['time_penalty'] = -time_penalty
        
        # Log detailed reward breakdown (debug level)
        self.logger.debug(f"Reward breakdown: {reward_details}, total: {reward:.3f}")
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment
        Nothing to do here as rendering is already handled by PyBullet GUI
        """
        pass
    
    def close(self):
        """
        Clean up the environment
        """
        if self.env:
            self.logger.info("Closing environment")
            self.env.close()
            self.env = None