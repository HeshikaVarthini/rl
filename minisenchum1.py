import numpy as np
import pybullet as p
from pybullet_envs.bullet import minitaur_gym_env
from gym import spaces
'''

class MinitaurWithSensors(minitaur_gym_env.MinitaurBulletEnv):
    def __init__(self, render_mode=None):
        super().__init__(render=(render_mode == "human"))

        self.num_sensors = 8
        self.sensor_range = 2.0  # meters
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)

        self.robot_id = None
        self.sensor_lines = []

    def reset(self):
        obs = super().reset()

        # Cache robot ID after reset
        if hasattr(self, 'robot') and hasattr(self.robot, 'quadruped'):
            self.robot_id = self.robot.quadruped
            sensor_readings = self._get_distance_sensor_readings()
            obs = np.concatenate([obs, sensor_readings])

            # Set observation space if not already done
            if not hasattr(self, 'observation_space') or self.observation_space is None:
                total_dim = obs.shape[0]
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        else:
            print("Warning: robot or quadruped not initialized yet.")

        # Optional: better camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2]
        )

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.robot_id is not None:
            sensor_readings = self._get_distance_sensor_readings()
            obs = np.concatenate([obs, sensor_readings])
        else:
            print("Warning: robot_id not available in step().")

        return obs, reward, done, info

    def _get_distance_sensor_readings(self):
        readings = []
        if self.robot_id is None:
            return np.zeros(self.num_sensors, dtype=np.float32)

        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_mat = p.getMatrixFromQuaternion(base_ori)
        forward_vec = np.array([base_mat[0], base_mat[3], base_mat[6]])
        right_vec = np.array([base_mat[1], base_mat[4], base_mat[7]])
        up_vec = np.array([base_mat[2], base_mat[5], base_mat[8]])

        # Clear previous debug lines
        for line_id in self.sensor_lines:
            p.removeUserDebugItem(line_id)
        self.sensor_lines = []

        for angle in self.sensor_angles:
            dir_vec = (
                np.cos(angle) * forward_vec +
                np.sin(angle) * right_vec
            )
            from_pos = np.array(base_pos) + 0.1 * up_vec
            to_pos = from_pos + self.sensor_range * dir_vec

            result = p.rayTest(from_pos.tolist(), to_pos.tolist())[0]
            hit_fraction = result[2]
            hit_object_uid = result[0]

            distance = hit_fraction * self.sensor_range
            readings.append(distance)

            # Draw to hit point
            hit_pos = from_pos + distance * dir_vec
            color = [0, 1, 0] if hit_fraction == 1.0 else [1, 0, 0]  # green if no hit, red if hit
            line_id = p.addUserDebugLine(from_pos, hit_pos, lineColorRGB=color, lineWidth=2.0, lifeTime=0.1)
            self.sensor_lines.append(line_id)

        return np.array(readings, dtype=np.float32)
'''
'''import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time

class MinitaurWithSensors(gym.Env):
    def __init__(self, render_mode=None):
        super(MinitaurWithSensors, self).__init__()
        self.render_mode = render_mode
        self.num_sensors = 8
        self.sensor_range = 2.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.sensor_lines = []

        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.3]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("quadruped/minitaur.urdf", start_pos, start_ori)

        obs_dim = 33 + self.num_sensors
        act_dim = p.getNumJoints(self.robot_id)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(act_dim,), dtype=np.float32)

    def step(self, action):
        for i in range(len(action)):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=action[i])

        p.stepSimulation()
        time.sleep(1./240.)

        obs = self._get_obs()
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.3]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("quadruped/minitaur.urdf", start_pos, start_ori)
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, list(range(p.getNumJoints(self.robot_id))))
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        sensor_readings = self._get_distance_sensor_readings()
        obs = np.concatenate([joint_positions, joint_velocities, base_pos, base_ori, base_vel, base_ang_vel, sensor_readings])
        return obs

    def _get_distance_sensor_readings(self):
        readings = []
        if self.robot_id is None:
            return np.zeros(self.num_sensors, dtype=np.float32)

        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_mat = p.getMatrixFromQuaternion(base_ori)
        forward_vec = np.array([base_mat[0], base_mat[3], base_mat[6]])
        right_vec = np.array([base_mat[1], base_mat[4], base_mat[7]])
        up_vec = np.array([base_mat[2], base_mat[5], base_mat[8]])

        ray_start_offset = 0.2
        from_pos = np.array(base_pos) + ray_start_offset * up_vec

        for line_id in self.sensor_lines:
            p.removeUserDebugItem(line_id)
        self.sensor_lines = []

        for angle in self.sensor_angles:
            dir_vec = np.cos(angle) * forward_vec + np.sin(angle) * right_vec
            to_pos = from_pos + self.sensor_range * dir_vec
            result = p.rayTest(from_pos.tolist(), to_pos.tolist())[0]
            hit_fraction = result[2]
            distance = hit_fraction * self.sensor_range
            readings.append(distance)

            color = [0, 1, 0] if hit_fraction == 1.0 else [1, 0, 0]
            hit_point = from_pos + distance * dir_vec
            line_id = p.addUserDebugLine(from_pos, hit_point, lineColorRGB=color, lineWidth=2.5, lifeTime=0.2)
            self.sensor_lines.append(line_id)

        return np.array(readings, dtype=np.float32)

if __name__ == "__main__":
    env = MinitaurWithSensors(render_mode="human")
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()
'''
'''
#main..detect obj
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time
import os

from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

class MinitaurWithObstacleEnv(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 2.0
        self.sensor_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_sensors, endpoint=True)
        self.sensor_lines = []
        self.obstacle_id = None
        
        super(MinitaurWithObstacleEnv, self).__init__(render=render)

        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)

    def reset(self):
        super(MinitaurWithObstacleEnv, self).reset()
        
        obstacle_position = [1.5, 0, 0.5]
        obstacle_urdf_path = os.path.join(pybullet_data.getDataPath(), "cube.urdf")
        self.obstacle_id = self._pybullet_client.loadURDF(
            obstacle_urdf_path,
            obstacle_position,
            useFixedBase=True
        )

        return self._get_full_observation()

    def step(self, action):
        parent_obs, reward, done, info = super(MinitaurWithObstacleEnv, self).step(action)
        
        contact_points = self._pybullet_client.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True

        sensor_readings = self._get_distance_sensor_readings()
        min_distance = np.min(sensor_readings)
        
        if not done and min_distance < 0.5:
            reward -= (0.5 - min_distance) * 20

        full_obs = np.concatenate([parent_obs, sensor_readings])
        
        return full_obs, reward, done, info

    def _get_full_observation(self):
        parent_obs = super(MinitaurWithObstacleEnv, self)._get_observation()
        sensor_readings = self._get_distance_sensor_readings()
        return np.concatenate([parent_obs, sensor_readings])

    def _get_distance_sensor_readings(self):
        """
        Performs ray tests, now correctly ignoring the floor.
        """
        readings = []
        base_pos, base_ori = self._pybullet_client.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_mat = self._pybullet_client.getMatrixFromQuaternion(base_ori)
        
        forward_vec = np.array([base_mat[0], base_mat[1], base_mat[2]])
        right_vec = np.array([base_mat[3], base_mat[4], base_mat[5]])
        up_vec = np.array([base_mat[6], base_mat[7], base_mat[8]])

        ray_start_offset = 0.1
        from_pos = np.array(base_pos) + ray_start_offset * up_vec
        
        for line_id in self.sensor_lines:
            self._pybullet_client.removeUserDebugItem(line_id)
        self.sensor_lines = []

        for angle in self.sensor_angles:
            dir_vec = np.cos(angle) * forward_vec + np.sin(angle) * right_vec
            to_pos = np.array(from_pos) + self.sensor_range * dir_vec
            
            result = self._pybullet_client.rayTest(from_pos, to_pos)[0]
            
            hit_body_id = result[0]
            hit_fraction = result[2]
            
            # --- FIX: Check if the ray hit the ground plane ---
            # The ground plane is the first object loaded, so its body ID is 0.
            if hit_body_id == 0:
                # If the ray hit the ground, treat it as a miss (full range)
                hit_fraction = 1.0

            distance = hit_fraction * self.sensor_range
            readings.append(distance)
            
            if self._is_render:
                color = [0, 1, 0] if hit_fraction == 1.0 else [1, 0, 0]
                hit_point = np.array(from_pos) + distance * dir_vec
                line_id = self._pybullet_client.addUserDebugLine(from_pos, hit_point.tolist(), lineColorRGB=color, lineWidth=1.5, lifeTime=1)
                self.sensor_lines.append(line_id)

        return np.array(readings, dtype=np.float32)

if __name__ == "__main__":
    env = MinitaurWithObstacleEnv(render=True)
    obs = env.reset()
    
    print("Environment is running. The robot will move randomly.")
    print("The sensor lines should now ignore the floor.")

    for i in range(1000):
        action = env.action_space.sample() 
        obs, reward, done, info = env.step(action)
        
        if i % 100 == 0:
            print(f"Step: {i}, Reward: {reward:.2f}, Done: {done}")
        
        if done:
            print("Episode finished! Resetting...")
            obs = env.reset()
            
    env.close()'''
'''
#implement and turn
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time
import os

from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

class MinitaurWithObstacleEnv(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 2.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.sensor_lines = []
        self.obstacle_id = None
        self.gait_phase = 0
        
        super(MinitaurWithObstacleEnv, self).__init__(render=render)

        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)

    def reset(self):
        obs = super(MinitaurWithObstacleEnv, self).reset()
        
        obstacle_position = [1.5, 0, 0.5]
        obstacle_urdf_path = os.path.join(pybullet_data.getDataPath(), "cube.urdf")
        
        self.obstacle_id = self._pybullet_client.loadURDF(
            obstacle_urdf_path,
            obstacle_position,
            useFixedBase=True,
            globalScaling=0.5
        )
        self.gait_phase = 0

        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2]
        )
        
        return self._get_full_observation()

    def _get_forward_action(self):
        """
        --- FIX: Replaced static push with a more effective two-phase diagonal crawl. ---
        """
        hip_push = -1.0 # Pushes hips backward for forward motion
        knee_up = 0.5   # Lifts the leg
        knee_down = -0.5 # Pushes the leg down

        if self.gait_phase % 2 == 0:  # Push with front-right and back-left
            action = np.array([
                hip_push, knee_down,    # Front right (pushing)
                0, knee_up,             # Front left (lifting)
                0, knee_up,             # Back right (lifting)
                hip_push, knee_down     # Back left (pushing)
            ])
        else:  # Push with front-left and back-right
            action = np.array([
                0, knee_up,             # Front right (lifting)
                hip_push, knee_down,    # Front left (pushing)
                hip_push, knee_down,    # Back right (pushing)
                0, knee_up              # Back left (lifting)
            ])
        
        self.gait_phase += 1
        return action

    def _get_turn_action(self, direction="right"):
        turn_amplitude = 1.0
        if direction == "right":
            turn_action = np.array([
                -turn_amplitude, 0,
                turn_amplitude, 0,
                -turn_amplitude, 0,
                turn_amplitude, 0
            ])
        else:
            turn_action = np.array([
                turn_amplitude, 0,
                -turn_amplitude, 0,
                turn_amplitude, 0,
                -turn_amplitude, 0
            ])
        return turn_action

    def step(self, action):
        parent_obs, reward, done, info = super(MinitaurWithObstacleEnv, self).step(action)
        
        contact_points = self._pybullet_client.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True

        sensor_readings = self._get_distance_sensor_readings()
        min_distance = np.min(sensor_readings)
        
        if not done and min_distance < 0.5:
            reward -= (0.5 - min_distance) * 20

        full_obs = np.concatenate([parent_obs, sensor_readings])
        
        return full_obs, reward, done, info

    def _get_full_observation(self):
        parent_obs = super(MinitaurWithObstacleEnv, self)._get_observation()
        sensor_readings = self._get_distance_sensor_readings()
        return np.concatenate([parent_obs, sensor_readings])

    def _get_distance_sensor_readings(self):
        readings = []
        base_pos, base_ori = self._pybullet_client.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_mat = self._pybullet_client.getMatrixFromQuaternion(base_ori)
        
        forward_vec = np.array([base_mat[0], base_mat[1], base_mat[2]])
        right_vec = np.array([base_mat[3], base_mat[4], base_mat[5]])
        up_vec = np.array([base_mat[6], base_mat[7], base_mat[8]])

        ray_start_offset = 0.1
        from_pos = np.array(base_pos) + ray_start_offset * up_vec
        
        for line_id in self.sensor_lines:
            self._pybullet_client.removeUserDebugItem(line_id)
        self.sensor_lines = []

        for angle in self.sensor_angles:
            dir_vec = np.cos(angle) * forward_vec + np.sin(angle) * right_vec
            to_pos = np.array(from_pos) + self.sensor_range * dir_vec
            
            result = self._pybullet_client.rayTest(from_pos.tolist(), to_pos.tolist())[0]
            
            hit_body_id = result[0]
            hit_fraction = result[2]
            
            if hit_body_id == 0:
                hit_fraction = 1.0

            distance = hit_fraction * self.sensor_range
            readings.append(distance)
            
            if self._is_render:
                color = [0, 1, 0] if hit_fraction == 1.0 else [1, 0, 0]
                hit_point = np.array(from_pos) + distance * dir_vec
                line_id = self._pybullet_client.addUserDebugLine(from_pos.tolist(), hit_point.tolist(), lineColorRGB=color, lineWidth=2.5, lifeTime=0.2)
                self.sensor_lines.append(line_id)

        return np.array(readings, dtype=np.float32)

if __name__ == "__main__":
    env = MinitaurWithObstacleEnv(render=True)
    obs = env.reset()
    
    print("Environment is running. The robot will now crawl forward and avoid the obstacle.")
    
    turning_counter = 0
    turning_direction = None

    for i in range(1000):
        front_sensor_readings = np.concatenate([obs[19:22], obs[22:25]])
        
        obstacle_on_left = np.any(front_sensor_readings[:3] < 1.5)
        obstacle_on_right = np.any(front_sensor_readings[3:] < 1.5)
        
        if turning_counter > 0:
            action = env._get_turn_action(direction=turning_direction)
            turning_counter -= 1
        elif obstacle_on_left:
            turning_direction = "right"
            turning_counter = 50
            action = env._get_turn_action(direction=turning_direction)
        elif obstacle_on_right:
            turning_direction = "left"
            turning_counter = 50
            action = env._get_turn_action(direction=turning_direction)
        else:
            action = env._get_forward_action()
            
        obs, reward, done, info = env.step(action)
        
        if i % 100 == 0:
            print(f"Step: {i}, Reward: {reward:.2f}, Done: {done}")
        
        if done:
            print("Episode finished! Resetting...")
            obs = env.reset()
            turning_counter = 0
            
    env.close()'''
'''import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time
import os
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 2.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.sensor_lines = []
        self.obstacle_id = None
        self.gait_phase = 0
        
        super().__init__(render=render)
        
        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)
        
    def reset(self):
        obs = super().reset()
        
        obstacle_position = [1.5, 0, 0.5]
        obstacle_urdf_path = os.path.join(pybullet_data.getDataPath(), "cube.urdf")
        
        self.obstacle_id = self._pybullet_client.loadURDF(
            obstacle_urdf_path,
            obstacle_position,
            useFixedBase=True,
            globalScaling=0.5
        )
        self.gait_phase = 0
        
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2]
        )
        
        return self._get_full_observation()

    def step(self, action):
        parent_obs, reward, done, info = super().step(action)
        
        contact_points = self._pybullet_client.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True
            
        sensor_readings = self._get_distance_sensor_readings()
        min_distance = np.min(sensor_readings)
        
        if not done and min_distance < 0.5:
            reward -= (0.5 - min_distance) * 20
            
        full_obs = np.concatenate([parent_obs, sensor_readings])
        
        return full_obs, reward, done, info

    def _get_distance_sensor_readings(self):
        readings = []
        base_pos, base_ori = self._pybullet_client.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_mat = self._pybullet_client.getMatrixFromQuaternion(base_ori)
        
        forward_vec = np.array([base_mat[0], base_mat[1], base_mat[2]])
        right_vec = np.array([base_mat[3], base_mat[4], base_mat[5]])
        up_vec = np.array([base_mat[6], base_mat[7], base_mat[8]])

        ray_start_offset = 0.1
        from_pos = np.array(base_pos) + ray_start_offset * up_vec

        for line_id in self.sensor_lines:
            self._pybullet_client.removeUserDebugItem(line_id)
        self.sensor_lines = []

        for angle in self.sensor_angles:
            dir_vec = np.cos(angle) * forward_vec + np.sin(angle) * right_vec
            to_pos = from_pos + self.sensor_range * dir_vec

            result = self._pybullet_client.rayTest(from_pos.tolist(), to_pos.tolist())[0]
            hit_fraction = result[2]
            
            if hit_fraction == 1.0:
                distance = self.sensor_range
            else:
                distance = hit_fraction * self.sensor_range
                
            readings.append(distance)
            
            if self._is_render:
                hit_pos = from_pos + distance * dir_vec
                color = [0, 1, 0] if hit_fraction == 1.0 else [1, 0, 0]
                line_id = self._pybullet_client.addUserDebugLine(from_pos.tolist(), hit_pos.tolist(), 
                                                                 lineColorRGB=color, lineWidth=2.0, lifeTime=0.1)
                self.sensor_lines.append(line_id)
        
        return np.array(readings, dtype=np.float32)

    def _get_full_observation(self):
        parent_obs = super()._get_observation()
        sensor_readings = self._get_distance_sensor_readings()
        return np.concatenate([parent_obs, sensor_readings])

    def _get_forward_action(self):
        """
        --- FIX: Corrected hip control signs for proper forward motion ---
        Joint order: [FL_hip, FL_knee, FR_hip, FR_knee, BL_hip, BL_knee, BR_hip, BR_knee]
        """
        # Hip values are now positive for backward push and negative for forward swing
        hip_push = 0.8    
        hip_swing = -0.4    
        knee_push = 0.4   
        knee_lift = -0.4   
        
        if self.gait_phase % 2 == 0:  # Push with front-right and back-left
            action = np.array([
                hip_swing, knee_lift,           # Front left (lifting and swinging)
                hip_push, knee_push,            # Front right (pushing down)
                hip_push, knee_push,            # Back left (pushing down)
                hip_swing, knee_lift            # Back right (lifting and swinging)
            ])
        else:  # Push with front-left and back-right
            action = np.array([
                hip_push, knee_push,            # Front left (pushing down)
                hip_swing, knee_lift,           # Front right (lifting and swinging)
                hip_swing, knee_lift,           # Back left (lifting and swinging)
                hip_push, knee_push             # Back right (pushing down)
            ])
        
        self.gait_phase += 1
        return action
    
    def _get_turn_action(self, direction="right"):
        """
        --- FIX: Corrected hip control signs for the turning gait ---
        Joint order: [FL_hip, FL_knee, FR_hip, FR_knee, BL_hip, BL_knee, BR_hip, BR_knee]
        """
        turn_amplitude = 0.8
        knee_lift = -0.5 
        
        if direction == "right":
            # Right turn: right legs push backward (positive), left legs swing forward (negative)
            action = np.array([
                -turn_amplitude, knee_lift,    
                turn_amplitude, 0,           
                -turn_amplitude, 0,            
                turn_amplitude, knee_lift    
            ])
        else:  # Left turn
            # Left turn: left legs push backward (positive), right legs swing forward (negative)
            action = np.array([
                turn_amplitude, 0,           
                -turn_amplitude, knee_lift,    
                turn_amplitude, knee_lift,   
                -turn_amplitude, 0             
            ])
        return action

if __name__ == "__main__":
    env = MinitaurWithSensors(render=True)
    obs = env.reset()
    
    print("Environment is running. The robot will now crawl forward and avoid the obstacle.")
    
    turning_counter = 0
    turning_direction = None
    
    for i in range(1000):
        sensor_readings = obs[-8:]
        
        front_left = sensor_readings[3]
        front = sensor_readings[4]
        front_right = sensor_readings[5]
        
        obstacle_on_left = (front_left < 1.2) or (front < 1.0 and front_left < front_right)
        obstacle_on_right = (front_right < 1.2) or (front < 1.0 and front_right < front_left)
        
        if turning_counter > 0:
            action = env._get_turn_action(direction=turning_direction)
            turning_counter -= 1
        elif obstacle_on_left:
            turning_direction = "right"
            turning_counter = 80
            action = env._get_turn_action(direction=turning_direction)
        elif obstacle_on_right:
            turning_direction = "left"
            turning_counter = 80
            action = env._get_turn_action(direction=turning_direction)
        else:
            if turning_counter == 0 and turning_direction is not None:
                env.gait_phase = 0
                turning_direction = None
                print("Resetting gait phase and resuming forward motion.")

            action = env._get_forward_action()
            
        obs, reward, done, info = env.step(action)
        
        if i % 100 == 0:
            print(f"Step: {i}, Reward: {reward:.2f}, Done: {done}")
        
        if done:
            print("Episode finished! Resetting...")
            obs = env.reset()
            turning_counter = 0
            
    env.close()'''
'''import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time
import os
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv


class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 2.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.sensor_lines = []
        self.obstacle_id = None
        self.gait_phase = 0
        self._is_render = render

        super().__init__(render=render)

        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)

    def reset(self):
        obs = super().reset()

        obstacle_position = [1.5, 0, 0.5]
        obstacle_urdf_path = os.path.join(pybullet_data.getDataPath(), "cube.urdf")

        self.obstacle_id = self._pybullet_client.loadURDF(
            obstacle_urdf_path,
            obstacle_position,
            useFixedBase=True,
            globalScaling=0.3  # smaller for easier debugging
        )
        self.gait_phase = 0

        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2]
        )

        return self._get_full_observation()

    def step(self, action):
        parent_obs, reward, done, info = super().step(action)

        contact_points = self._pybullet_client.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True

        sensor_readings = self._get_distance_sensor_readings()
        min_distance = np.min(sensor_readings)

        if not done and min_distance < 0.5:
            reward -= (0.5 - min_distance) * 20

        full_obs = np.concatenate([parent_obs, sensor_readings])

        return full_obs, reward, done, info

    def _get_distance_sensor_readings(self):
        readings = []
        base_pos, base_ori = self._pybullet_client.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_mat = self._pybullet_client.getMatrixFromQuaternion(base_ori)

        forward_vec = np.array([base_mat[0], base_mat[1], base_mat[2]])
        right_vec = np.array([base_mat[3], base_mat[4], base_mat[5]])
        up_vec = np.array([base_mat[6], base_mat[7], base_mat[8]])

        ray_start_offset = 0.1
        from_pos = np.array(base_pos) + ray_start_offset * up_vec

        for line_id in self.sensor_lines:
            self._pybullet_client.removeUserDebugItem(line_id)
        self.sensor_lines = []

        for angle in self.sensor_angles:
            dir_vec = np.cos(angle) * forward_vec + np.sin(angle) * right_vec
            to_pos = from_pos + self.sensor_range * dir_vec

            result = self._pybullet_client.rayTest(from_pos.tolist(), to_pos.tolist())[0]
            hit_fraction = result[2]

            if hit_fraction == 1.0:
                distance = self.sensor_range
            else:
                distance = hit_fraction * self.sensor_range

            readings.append(distance)

            if self._is_render:
                hit_pos = from_pos + distance * dir_vec
                color = [0, 1, 0] if hit_fraction == 1.0 else [1, 0, 0]
                line_id = self._pybullet_client.addUserDebugLine(from_pos.tolist(), hit_pos.tolist(),
                                                                 lineColorRGB=color, lineWidth=2.0, lifeTime=0.1)
                self.sensor_lines.append(line_id)

        return np.array(readings, dtype=np.float32)

    def _get_full_observation(self):
        parent_obs = super()._get_observation()
        sensor_readings = self._get_distance_sensor_readings()
        return np.concatenate([parent_obs, sensor_readings])

    def _get_forward_action(self):
        """
        Corrected alternating gait for forward crawling.
        Joint order: [FL_hip, FL_knee, FR_hip, FR_knee, BL_hip, BL_knee, BR_hip, BR_knee]
        """
        hip_push = 0.6     # backward push
        hip_swing = -0.4   # forward swing
        knee_push = 1.0
        knee_lift = -0.6

        if self.gait_phase % 2 == 0:
            action = np.array([
                hip_swing, knee_lift,     # FL - swing
                hip_push,  knee_push,     # FR - push
                hip_push,  knee_push,     # BL - push
                hip_swing, knee_lift      # BR - swing
            ])
        else:
            action = np.array([
                hip_push,  knee_push,     # FL - push
                hip_swing, knee_lift,     # FR - swing
                hip_swing, knee_lift,     # BL - swing
                hip_push,  knee_push      # BR - push
            ])

        self.gait_phase += 1
        return action

    def _get_turn_action(self, direction="right"):
        """
        Turn in place: left or right.
        """
        turn_amplitude = 0.5
        knee_lift = -0.5

        if direction == "right":
            action = np.array([
                -turn_amplitude, knee_lift,
                 turn_amplitude, 0,
                -turn_amplitude, 0,
                 turn_amplitude, knee_lift
            ])
        else:  # Left turn
            action = np.array([
                 turn_amplitude, 0,
                -turn_amplitude, knee_lift,
                 turn_amplitude, knee_lift,
                -turn_amplitude, 0
            ])
        return action


if __name__ == "__main__":
    env = MinitaurWithSensors(render=True)
    obs = env.reset()

    print("Robot crawling forward with obstacle avoidance.")

    turning_counter = 0
    turning_direction = None

    for i in range(4000):
        sensor_readings = obs[-8:]

        front_left = sensor_readings[3]
        front = sensor_readings[4]
        front_right = sensor_readings[5]

        obstacle_on_left = (front_left < 1.2) or (front < 1.0 and front_left < front_right)
        obstacle_on_right = (front_right < 1.2) or (front < 1.0 and front_right < front_left)

        if turning_counter > 0:
            action = env._get_turn_action(direction=turning_direction)
            turning_counter -= 1
        elif obstacle_on_left:
            turning_direction = "right"
            turning_counter = 60
            action = env._get_turn_action(direction="right")
        elif obstacle_on_right:
            turning_direction = "left"
            turning_counter = 60
            action = env._get_turn_action(direction="left")
        else:
            if turning_counter == 0 and turning_direction is not None:
                env.gait_phase = 0
                turning_direction = None
                print("Resuming forward crawling...")

            action = env._get_forward_action()

        obs, reward, done, info = env.step(action)
        time.sleep(1. / 40)  # control timing

        if i % 100 == 0:
            print(f"Step {i}, Reward: {reward:.2f}, Done: {done}")

        if done:
            print("Episode ended. Resetting...")
            obs = env.reset()
            turning_counter = 0

    env.close()'''
'''
import time
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv


class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.step_counter = 0
        self.num_sensors = 8
        self.sensor_range = 2.0
        self.sensor_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_sensors)
        super(MinitaurWithSensors, self).__init__(render=(render_mode == "human"))

    def reset(self):
        self.step_counter = 0
        return super().reset()

    def _get_forward_gait_action(self):
        """
        Simulate a trot-like forward gait using sine waves with diagonal leg pairs in sync.
        Ensures values stay within [-1, 1] range.
        """
        t = self.step_counter / 20.0  # time scaled for smoothness
        amplitude = 0.8
        frequency = 2 * np.pi  # 1 Hz

        # Diagonal leg phase: FL+RR in phase, FR+RL opposite
        fl_rr = amplitude * np.sin(frequency * t)
        fr_rl = amplitude * np.sin(frequency * t + np.pi)

        # Hip and knee of each leg move together (same phase)
        action = np.array([
            fl_rr, fl_rr,  # FL
            fr_rl, fr_rl,  # FR
            fr_rl, fr_rl,  # RL
            fl_rr, fl_rr   # RR
        ])
        return action

    def step(self, action):
        self.step_counter += 1
        return super().step(action)


if __name__ == "__main__":
    env = MinitaurWithSensors(render_mode="human")
    obs = env.reset()

    for _ in range(4000):
        action = env._get_forward_gait_action()
        obs, reward, done, info = env.step(action)
        time.sleep(1. / 240)

        if done:
            obs = env.reset()
'''
'''
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 0.2
        self.sensor_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_sensors)
        self.obstacle_id = None
        super(MinitaurWithSensors, self).__init__(render=render)

    def reset(self):
        obs = super().reset()
        self._add_obstacle()
        return self._get_observation_with_sensors(obs)

    def _add_obstacle(self):
        if self.obstacle_id is not None:
            p.removeBody(self.obstacle_id)
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[3], base_matrix[6]])
        obstacle_position = np.array(base_position) + forward_vector * 1.0
        obstacle_position[2] = 0.1
        obstacle_urdf = pybullet_data.getDataPath() + "/cube_small.urdf"
        self.obstacle_id = p.loadURDF(obstacle_urdf, obstacle_position.tolist())

    def _get_observation_with_sensors(self, base_obs):
        sensor_data = []
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[3], base_matrix[6]])
        right_vector = np.array([base_matrix[1], base_matrix[4], base_matrix[7]])
        up_vector = np.array([base_matrix[2], base_matrix[5], base_matrix[8]])

        for angle in self.sensor_angles:
            ray_dir = (
                np.cos(angle) * forward_vector +
                np.sin(angle) * right_vector
            )
            ray_dir /= np.linalg.norm(ray_dir)
            start = np.array(base_position) + 0.1 * up_vector
            end = start + ray_dir * self.sensor_range
            result = p.rayTest(start, end)[0]
            hit_fraction = result[2] if result[0] != -1 else 1.0
            sensor_data.append(hit_fraction)

            color = [1, 0, 0] if result[0] != -1 else [0, 1, 0]
            shortened_end = start + ray_dir * self.sensor_range * hit_fraction
            p.addUserDebugLine(start, shortened_end, color, lifeTime=0.1, lineWidth=1)

        return np.concatenate([base_obs, sensor_data])

    def step(self, action):
        previous_x = p.getBasePositionAndOrientation(self.minitaur.quadruped)[0][0]
        base_obs, _, done, info = super().step(action)
        current_x = p.getBasePositionAndOrientation(self.minitaur.quadruped)[0][0]
        forward_reward = current_x - previous_x
        return self._get_observation_with_sensors(base_obs), forward_reward, done, info

if __name__ == '__main__':
    env = MinitaurWithSensors(render=True)
    obs = env.reset()
    done = False
    t = 0.0
    timestep = 1.0 / 240

    # Gait Parameters
    amplitude = 0.7
    frequency = 2.0  # Hz
    phase_offsets = [0, 0, np.pi, np.pi, 0, 0, np.pi, np.pi]  # front vs rear

    while not done:
        action = np.array([
            amplitude * np.sin(2 * np.pi * frequency * t + phase)
            for phase in phase_offsets
        ])
        obs, reward, done, _ = env.step(action)
        p.setTimeStep(timestep)
        p.stepSimulation()
        t += timestep'''

'''
#back walk
import time
import numpy as np
import pybullet as p
import pybullet_data
import os
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from gym import spaces

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render_mode=None):
        self.step_counter = 0
        self.num_sensors = 8
        self.sensor_range = 2.0  # meters
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.robot_id = None
        self.sensor_lines = []
        
        super().__init__(render=(render_mode == "human"))
        
        if hasattr(self, 'observation_space'):
            original_obs_dim = self.observation_space.shape[0]
            custom_obs_dim = original_obs_dim + self.num_sensors
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)

    def reset(self):
        self.step_counter = 0
        obs = super().reset()
        
        if hasattr(self, 'robot') and hasattr(self.robot, 'quadruped'):
            self.robot_id = self.robot.quadruped
            sensor_readings = self._get_distance_sensor_readings()
            obs = np.concatenate([obs, sensor_readings])
        else:
            print("Warning: Robot not fully initialized. Skipping sensor readings.")
        
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2]
        )
        return obs

    def step(self, action):
        self.step_counter += 1
        obs, reward, done, info = super().step(action)
        
        if self.robot_id is not None:
            sensor_readings = self._get_distance_sensor_readings()
            obs = np.concatenate([obs, sensor_readings])
        else:
            print("Warning: robot_id not available in step().")

        return obs, reward, done, info

    def _get_distance_sensor_readings(self):
        readings = []
        if self.robot_id is None:
            return np.zeros(self.num_sensors, dtype=np.float32)

        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_mat = p.getMatrixFromQuaternion(base_ori)
        forward_vec = np.array([base_mat[0], base_mat[1], base_mat[2]])
        right_vec = np.array([base_mat[3], base_mat[4], base_mat[5]])
        up_vec = np.array([base_mat[6], base_mat[7], base_mat[8]])

        for line_id in self.sensor_lines:
            p.removeUserDebugItem(line_id)
        self.sensor_lines = []
        
        from_pos = np.array(base_pos) + 0.1 * up_vec

        for angle in self.sensor_angles:
            dir_vec = np.cos(angle) * forward_vec + np.sin(angle) * right_vec
            to_pos = from_pos + self.sensor_range * dir_vec

            result = p.rayTest(from_pos.tolist(), to_pos.tolist())[0]
            hit_fraction = result[2]
            
            distance = hit_fraction * self.sensor_range
            readings.append(distance)

            hit_pos = from_pos + distance * dir_vec
            color = [0, 1, 0] if hit_fraction == 1.0 else [1, 0, 0]
            line_id = p.addUserDebugLine(from_pos, hit_pos, lineColorRGB=color, lineWidth=2.0, lifeTime=0.1)
            self.sensor_lines.append(line_id)

        return np.array(readings, dtype=np.float32)

    def _get_forward_gait_action(self):
        """
        --- FIX: Inverted hip signs to finally move forward ---
        Generates a forward-moving trot gait.
        """
        t = self.step_counter / 20.0
        
        # A positive hip value corresponds to a backward push, which is what we need for propulsion.
        hip_amplitude = 0.8
        hip_fl_rr = hip_amplitude * np.sin(2 * np.pi * t)
        hip_fr_rl = hip_amplitude * np.sin(2 * np.pi * t + np.pi)

        # Knee joints must be in phase with the hip for propulsion.
        knee_amplitude = 0.5
        knee_fl_rr = knee_amplitude * np.sin(2 * np.pi * t)
        knee_fr_rl = knee_amplitude * np.sin(2 * np.pi * t + np.pi)
        
        action = np.array([
            hip_fl_rr, knee_fl_rr, 
            hip_fr_rl, knee_fr_rl,
            hip_fr_rl, knee_fr_rl,
            hip_fl_rr, knee_fl_rr
        ])

        return np.clip(action, -1.0, 1.0)
    
    def _get_turn_gait_action(self, direction="right"):
        """
        Gait for turning left or right using a synchronized hip-knee approach.
        """
        t = self.step_counter / 20.0
        hip_amplitude = 0.8
        knee_amplitude = 0.5

        turn_sign = 1.0 if direction == "right" else -1.0
        
        # Hip action for turning. Left and right hips move opposite.
        hip_front_right_back_left = hip_amplitude * np.sin(2 * np.pi * t)
        hip_front_left_back_right = hip_amplitude * np.sin(2 * np.pi * t + np.pi)

        # Knee action for turning (synchronized with hip)
        knee_front_right_back_left = knee_amplitude * np.sin(2 * np.pi * t)
        knee_front_left_back_right = knee_amplitude * np.sin(2 * np.pi * t + np.pi)

        action = np.array([
            turn_sign * hip_front_left_back_right, knee_front_left_back_right, # FL
            -turn_sign * hip_front_right_back_left, knee_front_right_back_left, # FR
            -turn_sign * hip_front_right_back_left, knee_front_right_back_left,  # BL
            turn_sign * hip_front_left_back_right, knee_front_left_back_right     # BR
        ])
        
        return np.clip(action, -1.0, 1.0)

if __name__ == "__main__":
    env = MinitaurWithSensors(render_mode="human")
    obs = env.reset()

    print("Running a continuous walk and turn sequence.")
    
    for _ in range(4000):
        if env.step_counter % 400 < 300:
            # Walk forward for 300 steps
            action = env._get_forward_gait_action()
        else:
            # Turn right for 100 steps
            action = env._get_turn_gait_action(direction="right")
            
        obs, reward, done, info = env.step(action)
        time.sleep(1. / 240)

        if done:
            print("Episode finished! Resetting...")
            obs = env.reset()

    env.close()'''
'''
#walk sense detec
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
import time

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 1.0
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.obstacle_id = None
        self.step_counter = 0
        self.sensor_lines = []
        
        super().__init__(render=render)
        
        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)

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
        
        obstacle_position = np.array(base_position) + forward_vector * 3
        obstacle_position[2] = 0.1
        obstacle_urdf = pybullet_data.getDataPath() + "/cube.urdf"
        self.obstacle_id = p.loadURDF(obstacle_urdf, obstacle_position.tolist(), useFixedBase=True)

    def _get_observation_with_sensors(self, base_obs):
        sensor_data = []
        base_position, base_orientation = p.getBasePositionAndOrientation(self.minitaur.quadruped)
        base_matrix = p.getMatrixFromQuaternion(base_orientation)
        forward_vector = np.array([base_matrix[0], base_matrix[1], base_matrix[2]])
        right_vector = np.array([base_matrix[3], base_matrix[4], base_matrix[5]])
        
        # --- FIX: Changed 'base_mat' to 'base_matrix' ---
        up_vector = np.array([base_matrix[6], base_matrix[7], base_matrix[8]])
        
        for line_id in self.sensor_lines:
            p.removeUserDebugItem(line_id)
        self.sensor_lines = []

        for angle in self.sensor_angles:
            ray_dir = (
                np.cos(angle) * forward_vector +
                np.sin(angle) * right_vector
            )
            ray_dir /= np.linalg.norm(ray_dir)
            start = np.array(base_position) + 0.1 * up_vector
            end = start + ray_dir * self.sensor_range
            result = p.rayTest(start, end)[0]
            hit_fraction = result[2] if result[0] != -1 else 1.0
            sensor_data.append(hit_fraction)
            
            if self._is_render:
                color = [1, 0, 0] if result[0] != -1 else [0, 1, 0]
                shortened_end = start + ray_dir * self.sensor_range * hit_fraction
                line_id = p.addUserDebugLine(start, shortened_end, color, lifeTime=0.1, lineWidth=1)
                self.sensor_lines.append(line_id)

        return np.concatenate([base_obs, sensor_data])

    def step(self, action):
        self.step_counter += 1
        base_obs, reward, done, info = super().step(action)
        
        contact_points = p.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True
        
        return self._get_observation_with_sensors(base_obs), reward, done, info

    def _get_forward_gait_action(self):
        t = self.step_counter / 20.0
        amplitude = 0.7
        frequency = 2.0
        
        hip_knee_fl_bl = amplitude * np.sin(2 * np.pi * frequency * t)
        hip_knee_fr_br = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)
        
        action = np.array([
            hip_knee_fl_bl, hip_knee_fl_bl,
            hip_knee_fr_br, hip_knee_fr_br,
            hip_knee_fl_bl, hip_knee_fl_bl,
            hip_knee_fr_br, hip_knee_fr_br
        ])
        
        return np.clip(action, -1.0, 1.0)
    
    def _get_turn_gait_action(self, direction="right"):
        t = self.step_counter / 20.0
        amplitude = 0.7
        frequency = 2.0
        
        turn_sign = 1.0 if direction == "right" else -1.0
        
        hip_knee_left = amplitude * np.sin(2 * np.pi * frequency * t)
        hip_knee_right = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)
        
        action = np.array([
            turn_sign * hip_knee_left, hip_knee_left,
            -turn_sign * hip_knee_right, hip_knee_right,
            turn_sign * hip_knee_left, hip_knee_left,
            -turn_sign * hip_knee_right, hip_knee_right
        ])
        
        return np.clip(action, -1.0, 1.0)

if __name__ == '__main__':
    env = MinitaurWithSensors(render=True)
    obs = env.reset()
    done = False
    
    print("Running with obstacle avoidance. The robot will walk forward and turn to avoid the cube.")
    
    while not done:
        sensor_readings = obs[-8:]
        front_sensor_index = np.where(np.isclose(env.sensor_angles, 0, atol=0.1))[0]
        front_sensor = sensor_readings[front_sensor_index].mean()
        
        if front_sensor < 0.5:
            action = env._get_turn_gait_action(direction="left")
            print("Obstacle detected! Turning left.")
        else:
            action = env._get_forward_gait_action()
            
        obs, reward, done, _ = env.step(action)
        time.sleep(1.0 / 240)

    env.close()'''
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
import time

class MinitaurWithSensors(MinitaurBulletEnv):
    def __init__(self, render=False):
        self.num_sensors = 8
        self.sensor_range = 3.0  # <--- CHANGED FROM 2.0 TO 3.0
        # Angles in radians: [-pi, -3pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3pi/4]
        # These correspond to Back, Back-left, Left, Front-left, Front, Front-right, Right, Back-right
        self.sensor_angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        self.obstacle_id = None
        self.step_counter = 0
        self.sensor_lines = []
        self.max_episode_steps = 1000  # ← CRITICAL: Limit episodes to 1000 steps max
        super().__init__(render=render)
        original_obs_dim = self.observation_space.shape[0]
        custom_obs_dim = original_obs_dim + self.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(custom_obs_dim,), dtype=np.float32)

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
            globalScaling=0.3
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
        sensor_height = 0.2
        start = np.array(base_position) + sensor_height * up_vector
        for angle in self.sensor_angles:
            ray_dir = (
                np.cos(angle) * forward_vector +
                np.sin(angle) * right_vector
            )
            ray_dir /= np.linalg.norm(ray_dir)
            end = start + ray_dir * self.sensor_range
            result = p.rayTest(start, end)[0]
            hit_object_id = result[0]
            if hit_object_id == self.obstacle_id:
                hit_fraction = result[2]
            else:
                hit_fraction = 1.0
            sensor_data.append(hit_fraction)
            if self._is_render:
                color = [1, 0, 0] if hit_object_id == self.obstacle_id else [0, 1, 0]
                shortened_end = start + ray_dir * self.sensor_range * hit_fraction
                line_id = p.addUserDebugLine(
                    start, shortened_end, color,
                    lifeTime=0.1,
                    lineWidth=3
                )
                self.sensor_lines.append(line_id)
        return np.concatenate([base_obs, sensor_data])

    def step(self, action):
        self.step_counter += 1
        base_obs, reward, done, info = super().step(action)
        
        # Check for collision with obstacle
        contact_points = p.getContactPoints(self.minitaur.quadruped, self.obstacle_id)
        if len(contact_points) > 0:
            reward -= 100.0
            done = True
        
        # ← CRITICAL FIX: Force episode to end after max steps
        # Without this, episodes run for 10,000+ steps and you only get 1 episode!
        if self.step_counter >= self.max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        
        return self._get_observation_with_sensors(base_obs), reward, done, info

    def _get_forward_gait_action(self):
        t = self.step_counter / 20.0
        amplitude = 0.7
        frequency = 2.0
        hip_knee_fl_bl = amplitude * np.sin(2 * np.pi * frequency * t)
        hip_knee_fr_br = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)
        action = np.array([
            hip_knee_fl_bl, hip_knee_fl_bl,
            hip_knee_fr_br, hip_knee_fr_br,
            hip_knee_fl_bl, hip_knee_fl_bl,
            hip_knee_fr_br, hip_knee_fr_br
        ])
        return np.clip(action, -1.0, 1.0)

    def _get_turn_gait_action(self, direction="right"):
        t = self.step_counter / 20.0
        amplitude = 0.7
        frequency = 2.0
        turn_amp = 0.5
        
        hip_knee_fl_bl = amplitude * np.sin(2 * np.pi * frequency * t)
        hip_knee_fr_br = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)

        if direction == "right":
            fl_hip_offset, bl_hip_offset = turn_amp, turn_amp
            fr_hip_offset, br_hip_offset = -turn_amp, -turn_amp
        else: # "left"
            fl_hip_offset, bl_hip_offset = -turn_amp, -turn_amp
            fr_hip_offset, br_hip_offset = turn_amp, turn_amp

        action = np.array([
            hip_knee_fl_bl + fl_hip_offset, hip_knee_fl_bl,
            hip_knee_fr_br + fr_hip_offset, hip_knee_fr_br,
            hip_knee_fl_bl + bl_hip_offset, hip_knee_fl_bl,
            hip_knee_fr_br + br_hip_offset, hip_knee_fr_br
        ])
        return np.clip(action, -1.0, 1.0)

if __name__ == '__main__':
    env = MinitaurWithSensors(render=True)
    obs = env.reset()
    done = False

    print("Running: robot will walk forward and turn away if cube detected.")
    print("----------------------------------------------------------------")

    turning_steps = 0
    turn_direction = "left"
    
    # Store actions for better control
    forward_action = env._get_forward_gait_action()

    while not done:
        sensor_readings = obs[-8:]
        
        # CORRECTED SENSOR INDICES
        front_sensor = sensor_readings[4]
        left_sensor = sensor_readings[2]
        right_sensor = sensor_readings[6]
        
        obstacle_threshold = 0.8
        obstacle_detected = (
            front_sensor < obstacle_threshold or
            left_sensor < obstacle_threshold or
            right_sensor < obstacle_threshold
        )
        
        print(f"Front: {front_sensor:.2f}, Left: {left_sensor:.2f}, Right: {right_sensor:.2f} | Turning: {turning_steps > 0}")

        if obstacle_detected:
            if turning_steps == 0:
                turn_direction = "left" if left_sensor > right_sensor else "right"
                turning_steps = 200 # Increased turning steps for a clearer turn
                print(f"Obstacle detected! Initiating a {turn_direction} turn.")
            action = env._get_turn_gait_action(direction=turn_direction)
        elif turning_steps > 0:
            # Gradually blend the turning action back to the forward action
            turn_progress = (200 - turning_steps) / 200.0
            turn_action = env._get_turn_gait_action(direction=turn_direction)
            forward_action = env._get_forward_gait_action()
            action = turn_action * (1.0 - turn_progress) + forward_action * turn_progress
            
            turning_steps -= 1
            if turning_steps == 0:
                print("Finished turning, resuming forward gait.")
        else:
            action = env._get_forward_gait_action()

        obs, reward, done, _ = env.step(action)
        time.sleep(1.0 / 60)

    env.close()