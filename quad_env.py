'''import gym
import pybullet as p
import pybullet_data
import numpy as np
import os
from gym import spaces
import time

class MiniCheetahEnv(gym.Env):
    def __init__(self, render=False):
        super(MiniCheetahEnv, self).__init__()
        self.render_mode = render
        if self.render_mode:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1. / 240., physicsClientId=self.client)

        urdf_path = "C:/Users/heshi/Mini-Cheetah-ROS/cheetah_description/xacro/mini_cheetah.urdf"
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot = p.loadURDF(urdf_path, [0, 0, 0.3], useFixedBase=False, physicsClientId=self.client)

        self.joint_indices = []
        self.joint_names = []

        num_joints = p.getNumJoints(self.robot)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot, i)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)

        self.num_ctrl_joints = len(self.joint_indices)
        print(f"Controllable joints: {self.joint_names}")

        # Set friction for all legs
        for j in self.joint_indices:
            p.changeDynamics(self.robot, j, lateralFriction=2.0)

        self.initial_pose = [0.0 for _ in range(self.num_ctrl_joints)]

        obs_dim = 3 + 4 + 3 + 3 + self.num_ctrl_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_ctrl_joints,), dtype=np.float32)

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        urdf_path = "C:/Users/heshi/Mini-Cheetah-ROS/cheetah_description/xacro/mini_cheetah.urdf"
        self.robot = p.loadURDF(urdf_path, [0, 0, 0.3], useFixedBase=False, physicsClientId=self.client)

        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot, joint_idx, self.initial_pose[i])
            p.changeDynamics(self.robot, joint_idx, lateralFriction=2.0)

        return self._get_obs()

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        joint_angles = [p.getJointState(self.robot, idx)[0] for idx in self.joint_indices]
        return np.array(list(pos) + list(orn) + list(lin_vel) + list(ang_vel) + joint_angles, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[i],
                force=20
            )
        p.stepSimulation()
        obs = self._get_obs()

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, _ = p.getBaseVelocity(self.robot)
        z = pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        reward = lin_vel[0] - abs(roll) - abs(pitch) - abs(z - 0.3)
        done = z < 0.18 or abs(roll) > 0.5 or abs(pitch) > 0.5
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)

if __name__ == "__main__":
    env = MiniCheetahEnv(render=True)
    obs = env.reset()

    freq = 2.0         # Gait frequency in Hz
    hip_amp = 0.6      # Hip joint amplitude
    thigh_amp = 0.8    # Thigh joint amplitude
    calf_amp = 0.6     # Calf joint amplitude
    duration = 20      # Total time to run
    step_time = 1. / 240.

    joint_map = {name: idx for name, idx in zip(env.joint_names, env.joint_indices)}

    # Joint groupings (you can print env.joint_names to confirm naming)
    front_left = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint']
    front_right = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint']
    rear_left = ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
    rear_right = ['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

    t = 0
    while t < duration:
        action = np.zeros(env.num_ctrl_joints)
        phase = 2 * np.pi * freq * t

        # Diagonal pair 1: FL + RR
        for jname in front_left + rear_right:
            idx = env.joint_names.index(jname)
            if "hip" in jname:
                action[idx] = hip_amp * np.sin(phase)
            elif "thigh" in jname:
                action[idx] = thigh_amp * np.sin(phase + np.pi / 2)
            elif "calf" in jname:
                action[idx] = -calf_amp * np.sin(phase + np.pi / 2)

        # Diagonal pair 2: FR + RL (out of phase)
        for jname in front_right + rear_left:
            idx = env.joint_names.index(jname)
            if "hip" in jname:
                action[idx] = hip_amp * np.sin(phase + np.pi)
            elif "thigh" in jname:
                action[idx] = thigh_amp * np.sin(phase + np.pi + np.pi / 2)
            elif "calf" in jname:
                action[idx] = -calf_amp * np.sin(phase + np.pi + np.pi / 2)

        obs, reward, done, _ = env.step(action)
        base_pos, _ = p.getBasePositionAndOrientation(env.robot)
        print(f"t={t:.2f}s | Reward={reward:.2f} | Pos x={base_pos[0]:.2f} | Done={done}")
        time.sleep(step_time)
        t += step_time

        if done:
            print("Fallen. Resetting...")
            obs = env.reset()
            t = 0

    env.close()'''
'''import pybullet as p
import pybullet_data
import numpy as np
import os
import time

# --- Configuration Paths ---
# IMPORTANT: Ensure these paths are correct for your system.
# The Mini Cheetah URDF is usually found in the 'cheetah_description' package
# from the Mini-Cheetah-ROS repository.
MINICHEETAH_URDF_DIR = "C:/Users/heshi/Mini-Cheetah-ROS/cheetah_description/xacro"
MINICHEETAH_URDF_FILE = "mini_cheetah.urdf"
URDF_PATH = os.path.join(MINICHEETAH_URDF_DIR, MINICHEETAH_URDF_FILE)

class MiniCheetahGaitController:
    """
    Controls a simulated Mini Cheetah robot in PyBullet to perform a trotting gait.
    """
    def __init__(self, render=True):
        """
        Initializes the PyBullet simulation environment and loads the robot.

        Args:
            render (bool): If True, runs the simulation with a GUI; otherwise, in direct mode.
        """
        if render:
            self.client = p.connect(p.GUI)
            # Disable PyBullet's default debug GUI elements for a cleaner view
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1) # Enable shadows for better visuals
        else:
            self.client = p.connect(p.DIRECT)

        # Set up PyBullet physics environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Allows loading default PyBullet assets
        p.setGravity(0, 0, -9.81) # Standard gravity
        self.time_step = 1.0 / 240.0 # Simulation time step (240 Hz)
        p.setTimeStep(self.time_step)

        # Load the ground plane with adjusted friction
        self.plane = p.loadURDF("plane.urdf")
        # Increased friction to prevent slipping
        p.changeDynamics(self.plane, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)

        # Load the Mini Cheetah robot URDF
        if not os.path.exists(URDF_PATH):
            print(f"Error: Mini Cheetah URDF file not found at {URDF_PATH}.")
            print("Please ensure the 'mini_cheetah.urdf' file is in the specified directory.")
            p.disconnect()
            raise FileNotFoundError(f"URDF file not found: {URDF_PATH}")

        # Load the robot slightly above the ground to allow settling
        self.robot = p.loadURDF(URDF_PATH, [0, 0, 0.25], useFixedBase=False)
        # Set a good initial camera view for observation
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.2])

        # Initialize joint information
        self.joint_indices = []
        self.joint_names = []
        self.joint_name_to_index = {} # Map joint name strings to their integer indices

        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")

            # Only consider revolute (hinge) joints for control
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
                self.joint_name_to_index[joint_name] = i

            # Apply friction to all links, especially important for foot links
            p.changeDynamics(self.robot, i, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)

        print("Controllable joints:", self.joint_names)

        # Define the robot's initial standing pose (tuned for Mini Cheetah)
        # These angles are crucial for a stable initial stance.
        # Thigh joints are typically flexed forward (positive angle), calf joints backward (negative angle).
        self.standing_pose = {
            'FL_hip_joint': 0.0, 'FL_thigh_joint': 0.7, 'FL_calf_joint': -1.4, # Adjusted slightly
            'FR_hip_joint': 0.0, 'FR_thigh_joint': 0.7, 'FR_calf_joint': -1.4, # Adjusted slightly
            'RL_hip_joint': 0.0, 'RL_thigh_joint': 0.7, 'RL_calf_joint': -1.4, # Adjusted slightly
            'RR_hip_joint': 0.0, 'RR_thigh_joint': 0.7, 'RR_calf_joint': -1.4, # Adjusted slightly
        }

        # Reset joints to the standing pose
        for jname, angle in self.standing_pose.items():
            if jname in self.joint_name_to_index:
                p.resetJointState(self.robot, self.joint_name_to_index[jname], angle)
            else:
                print(f"Warning: Joint '{jname}' specified in standing_pose not found in robot URDF.")

        # Allow the robot to settle into its initial pose under gravity
        print("Robot settling into standing pose...")
        for _ in range(240): # Simulate for 1 second to allow stabilization
            p.stepSimulation()
            time.sleep(self.time_step)
        print("Settling complete. Starting gait.")

    def run(self, duration_sec=20):
        """
        Executes the trotting gait for a specified duration.

        Args:
            duration_sec (float): The total duration in seconds to run the gait.
        """
        # --- Gait Parameters ---
        # Further reduced gait frequency for more stability
        gait_frequency = 0.8 # Hz - how fast the gait cycle repeats (e.g., 0.8 cycles per second)
        # stride_length_x = 0.15 # Not directly used in this joint-space control, but conceptual
        stride_height = 0.08 # How high the foot lifts during the swing phase (conceptual)

        # Duty cycle: percentage of time a leg is in stance phase (on ground)
        # A duty cycle > 0.5 (e.g., 0.6) means more than two legs are on the ground
        # at any given time, providing better stability for a trot.
        stance_duty_cycle = 0.6

        # Joint amplitudes for gait (these are offsets from the standing pose)
        hip_amplitude = 0.05 # Lateral movement of hip (abduction/adduction)
        # Adjusted thigh and calf amplitudes for more controlled movement and better foot clearance
        thigh_amplitude = 0.2 # Forward/backward movement of thigh (flexion/extension)
        calf_amplitude = 0.4 # Knee flexion/extension for foot lift (increased slightly for better clearance)

        # Maximum velocity for joint motors to prevent overshooting and oscillations
        max_joint_velocity = 5.0 # Radians per second

        t = 0 # Simulation time counter
        while t < duration_sec:
            # Calculate the current phase of the overall gait cycle (0 to 1)
            gait_phase = (t * gait_frequency) % 1.0

            # --- Trotting Gait Logic ---
            # In a trot, diagonal pairs of legs move in sync.
            # Group 1: Front-Left (FL) and Rear-Right (RR)
            # Group 2: Front-Right (FR) and Rear-Left (RL)

            # Group 1 starts its swing phase at gait_phase = 0.
            # Group 2 starts its swing phase at gait_phase = 0.5 (half a cycle later).

            target_joint_angles = {} # Dictionary to store calculated target angles for this time step

            for jname, standing_angle in self.standing_pose.items():
                leg_name_prefix = jname.split('_')[0] # e.g., 'FL', 'FR', 'RL', 'RR'
                joint_type = jname.split('_')[1]     # e.g., 'hip', 'thigh', 'calf'

                # Determine the phase for the current leg
                current_leg_phase = gait_phase
                if leg_name_prefix in ['FR', 'RL']:
                    # Offset phase for Group 2 legs
                    current_leg_phase = (gait_phase + 0.5) % 1.0

                # Determine if the leg is in swing or stance phase
                # Swing phase duration: (1.0 - stance_duty_cycle) of the total cycle
                is_swing_phase = current_leg_phase < (1.0 - stance_duty_cycle)

                # Normalize phase within its current segment (swing or stance) to 0 to 1
                if is_swing_phase:
                    # Phase for swing segment (0 to 1)
                    phase_in_segment = current_leg_phase / (1.0 - stance_duty_cycle)
                else:
                    # Phase for stance segment (0 to 1)
                    phase_in_segment = (current_leg_phase - (1.0 - stance_duty_cycle)) / stance_duty_cycle

                # Calculate the target angle for the current joint based on its type and phase
                target_angle = standing_angle # Start with the standing pose angle

                if joint_type == 'hip':
                    # Hip joint (abduction/adduction) for lateral stability.
                    # A small continuous sine wave oscillation throughout the cycle.
                    target_angle += hip_amplitude * np.sin(2 * np.pi * current_leg_phase)
                elif joint_type == 'thigh':
                    # Thigh joint (flexion/extension) for forward/backward motion.
                    # Uses a cosine wave to smoothly transition between forward and backward positions.
                    # This makes the thigh angle go from (standing + amp) -> standing -> (standing - amp) -> standing -> (standing + amp).
                    # This corresponds to the leg moving from a forward position, sweeping back, and then swinging forward again.
                    target_angle += thigh_amplitude * np.cos(2 * np.pi * current_leg_phase)
                elif joint_type == 'calf':
                    # Calf joint (knee flexion/extension) primarily for lifting the foot.
                    if is_swing_phase:
                        # During swing, the calf flexes (angle becomes more negative) to lift the foot,
                        # then extends (angle becomes less negative) to prepare for landing.
                        # Subtracting this from standing_angle makes the angle more negative (flex).
                        target_angle = standing_angle - calf_amplitude * np.sin(np.pi * phase_in_segment)
                    else:
                        # During stance, the calf stays relatively extended, maintaining ground contact.
                        # No additional offset is applied, keeping it at the standing angle.
                        target_angle = standing_angle

                target_joint_angles[jname] = target_angle

            # Apply the calculated target angles to the robot's motors
            for jname, target_angle in target_joint_angles.items():
                if jname in self.joint_name_to_index:
                    joint_idx = self.joint_name_to_index[jname]
                    # Use POSITION_CONTROL to move the joint to the target angle
                    # Increased 'force' for stronger motor control, added maxVelocity for damping
                    p.setJointMotorControl2(self.robot, joint_idx,
                                            p.POSITION_CONTROL,
                                            targetPosition=target_angle,
                                            force=200, # Increased force
                                            maxVelocity=max_joint_velocity) # Added max velocity

            # Step the simulation forward
            p.stepSimulation()
            # Introduce a small delay to match the simulation time step for real-time visualization
            time.sleep(self.time_step)
            # Increment the simulation time
            t += self.time_step

# --- Main execution block ---
if __name__ == "__main__":
    # Create an instance of the controller with GUI rendering enabled
    controller = MiniCheetahGaitController(render=True)
    # Run the gait for 30 seconds
    controller.run(duration_sec=30)
    # Disconnect from the PyBullet physics server when done
    p.disconnect()'''
import pybullet as p
import pybullet_data
import numpy as np
import os
import time

# === Path to Mini Cheetah URDF ===
MINICHEETAH_URDF_DIR = "C:/Users/heshi/Mini-Cheetah-ROS/cheetah_description/xacro"
MINICHEETAH_URDF_FILE = "mini_cheetah.urdf"
URDF_PATH = os.path.join(MINICHEETAH_URDF_DIR, MINICHEETAH_URDF_FILE)

class MiniCheetahGaitController:
    def __init__(self, render=True):
        if render:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.1)

        if not os.path.exists(URDF_PATH):
            raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
        
        # Load robot slightly above ground to allow settling
        self.robot = p.loadURDF(URDF_PATH, [0, 0, 0.23], useFixedBase=False)

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.2])

        self.joint_indices = []
        self.joint_name_to_index = {}

        for i in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                joint_name = joint_info[1].decode("utf-8")
                self.joint_name_to_index[joint_name] = i
                p.changeDynamics(self.robot, i, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.1)

        self.standing_pose = {
            'FL_hip_joint': 0.0, 'FL_thigh_joint': 0.8, 'FL_calf_joint': -1.6,
            'FR_hip_joint': 0.0, 'FR_thigh_joint': 0.8, 'FR_calf_joint': -1.6,
            'RL_hip_joint': 0.0, 'RL_thigh_joint': 0.8, 'RL_calf_joint': -1.6,
            'RR_hip_joint': 0.0, 'RR_thigh_joint': 0.8, 'RR_calf_joint': -1.6,
        }

        for jname, angle in self.standing_pose.items():
            p.resetJointState(self.robot, self.joint_name_to_index[jname], angle)

        for _ in range(240):
            p.stepSimulation()
            time.sleep(self.time_step)

    def run(self, duration_sec=10):
        gait_frequency = 1.0
        stride_height = 0.05
        stance_duty_cycle = 0.65
        hip_amp = 0.05
        thigh_amp = 0.25
        calf_amp = 0.45
        max_joint_velocity = 5.0

        t = 0
        while t < duration_sec:
            gait_phase = (t * gait_frequency) % 1.0
            target_angles = {}

            for jname in self.standing_pose:
                leg_prefix = jname.split('_')[0]
                jtype = jname.split('_')[1]

                current_phase = gait_phase
                if leg_prefix in ['FR', 'RL']:
                    current_phase = (gait_phase + 0.5) % 1.0

                is_swing = current_phase < (1.0 - stance_duty_cycle)
                phase_segment = (current_phase / (1.0 - stance_duty_cycle)) if is_swing else ((current_phase - (1.0 - stance_duty_cycle)) / stance_duty_cycle)

                base_angle = self.standing_pose[jname]
                angle = base_angle

                if jtype == 'hip':
                    angle += hip_amp * np.sin(2 * np.pi * current_phase)
                elif jtype == 'thigh':
                    angle += thigh_amp * np.cos(2 * np.pi * current_phase)
                elif jtype == 'calf':
                    if is_swing:
                        angle = base_angle - calf_amp * np.sin(np.pi * phase_segment)
                    else:
                        angle = base_angle
                target_angles[jname] = angle

            for jname, target in target_angles.items():
                idx = self.joint_name_to_index[jname]
                p.setJointMotorControl2(self.robot, idx, p.POSITION_CONTROL,
                                        targetPosition=target,
                                        force=250,
                                        maxVelocity=max_joint_velocity)

            p.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step

# === Entry Point ===
if __name__ == "__main__":
    robot = MiniCheetahGaitController(render=True)
    robot.run(duration_sec=30)
    p.disconnect()

