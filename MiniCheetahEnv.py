import pybullet as p
import pybullet_data
import numpy as np
import os
import time

# === Path to Mini Cheetah URDF ===
# IMPORTANT: DOUBLE-CHECK THIS PATH!
# Ensure this path points to the 'mini_cheetah.urdf' file
# that you have MODIFIED as instructed (added <inertial> to 'trunk' link).
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
        # Increase friction for better grip during walking
        p.changeDynamics(self.plane, -1, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.1)

        if not os.path.exists(URDF_PATH):
            raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

        self.robot = None # Will be loaded/reloaded in reset_robot
        self.joint_indices = []
        self.joint_name_to_index = {}
        
        # === Tuned Standing Pose ===
        self.standing_pose = {
            'FL_hip_joint': 0.05, 'FL_thigh_joint': 0.9, 'FL_calf_joint': -1.9,  # Front Left
            'FR_hip_joint': -0.05, 'FR_thigh_joint': 0.9, 'FR_calf_joint': -1.9, # Front Right
            'RL_hip_joint': 0.05, 'RL_thigh_joint': 0.9, 'RL_calf_joint': -1.9,  # Rear Left
            'RR_hip_joint': -0.05, 'RR_thigh_joint': 0.9, 'RR_calf_joint': -1.9, # Rear Right
        }
        
        self.reset_robot() # Perform initial robot setup and settle

        # Camera setup - adjusted slightly for a better view of walking
        p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.2])

    def reset_robot(self):
        """Resets the robot to its initial standing pose and allows it to settle."""
        if self.robot is not None:
            p.removeBody(self.robot) # Remove existing robot if any

        initial_z_pos = 0.28 # Common good starting height
        self.robot = p.loadURDF(URDF_PATH, [0, 0, initial_z_pos], useFixedBase=False)
        p.resetBaseVelocity(self.robot, [0,0,0], [0,0,0]) # Ensure no initial velocity or angular velocity

        self.joint_indices = []
        self.joint_name_to_index = {}

        for i in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            if joint_info[2] == p.JOINT_REVOLUTE: # Only consider revolute joints for control
                self.joint_indices.append(i)
                joint_name = joint_info[1].decode("utf-8")
                self.joint_name_to_index[joint_name] = i
                # Apply friction to joints as well (if not defined in URDF dynamics tags)
                p.changeDynamics(self.robot, i, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.1)

        for jname, angle in self.standing_pose.items():
            idx = self.joint_name_to_index[jname]
            p.resetJointState(self.robot, idx, angle)
            # Set motors with zero force initially to allow natural settling under gravity
            p.setJointMotorControl2(self.robot, idx, p.POSITION_CONTROL, targetPosition=angle, force=0)

        print("Robot settling...")
        for _ in range(3 * 240): # 3 seconds settling time at 240 Hz
            p.stepSimulation()
            time.sleep(self.time_step)
        print("Settling complete.")

        for jname, angle in self.standing_pose.items():
            idx = self.joint_name_to_index[jname]
            p.setJointMotorControl2(self.robot, idx, p.POSITION_CONTROL,
                                    targetPosition=angle,
                                    force=250, # Sufficient force for Mini Cheetah's motors
                                    maxVelocity=8.0) # Max velocity for motor (allows quicker response)

    def is_fallen(self):
        """Checks if the robot has fallen (base too low or rolled over significantly)."""
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        
        if pos[2] < 0.08: # If base is lower than 8cm, consider it fallen
            print(f"Robot fell: Z position {pos[2]:.3f} is too low.")
            return True
        
        roll, pitch, yaw = p.getEulerFromQuaternion(ori)
        if abs(roll) > 1.2 or abs(pitch) > 1.2: 
            print(f"Robot fell: Roll {np.degrees(roll):.1f} deg or Pitch {np.degrees(pitch):.1f} deg is too high.")
            return True
        return False

    def run(self, duration_sec=600):
        # --- Gait Parameters (Optimized for synchronized trot walking) ---
        gait_frequency = 1.5        # Cycles per second (faster trot for more noticeable movement)
        stride_horizontal_amp = 0.15 # Total horizontal displacement of foot during one cycle (m)
        stride_vertical_amp = 0.08   # Max height foot lifts during swing (m)

        stance_duty_cycle = 0.65    # Percentage of cycle foot is on ground (e.g., 0.65 for trot)
        swing_duty_cycle = 1.0 - stance_duty_cycle # Derived

        # Joint amplitudes (experiment with these!)
        hip_abduction_amp = 0.1     # Side-to-side motion of hip
        thigh_swing_amp = 0.6       # Overall thigh swing (forward/backward)
        calf_swing_retraction = 1.0 # How much calf bends back during swing (large for clearance)
        calf_stance_extension = 0.2 # How much calf extends during stance for push

        max_joint_velocity = 12.0   # Allow joints to move faster for dynamic gait
        
        t = 0
        while t < duration_sec:
            if self.is_fallen():
                print(f"Robot detected as fallen at t={t:.2f}s. Resetting simulation...")
                self.reset_robot()
                t = 0
                continue

            # --- Trot Gait Phase Calculation ---
            # Legs move in diagonal pairs: (FL, RR) and (FR, RL)
            # The two pairs are 0.5 (half cycle) out of phase.
            
            # Global phase (0 to 1) for the entire gait cycle
            gait_phase_global = (t * gait_frequency) % 1.0

            # Define phase offsets for a trot gait
            # Group 0: Front-Left (FL), Rear-Right (RR)
            # Group 1: Front-Right (FR), Rear-Left (RL)
            leg_group_phase_offsets = {
                'FL': 0.0,
                'FR': 0.5,
                'RL': 0.5,
                'RR': 0.0,
            }

            target_angles = {}

            for jname in self.standing_pose:
                leg_prefix = jname.split('_')[0]
                jtype = jname.split('_')[1]

                # Calculate individual leg's phase within its own cycle (0 to 1)
                # This ensures synchronization within diagonal pairs
                current_leg_phase = (gait_phase_global + leg_group_phase_offsets[leg_prefix]) % 1.0

                # Determine if leg is in swing or stance phase
                is_swing = current_leg_phase < swing_duty_cycle
                
                # Normalize phase within the current segment (0 to 1)
                if is_swing:
                    phase_in_segment = current_leg_phase / swing_duty_cycle
                else:
                    phase_in_segment = (current_leg_phase - swing_duty_cycle) / stance_duty_cycle

                base_angle = self.standing_pose[jname]
                angle_offset = 0.0

                # --- Joint Angle Calculations for Synchronized Locomotion ---
                # These equations define the trajectories of the leg joints to achieve walking.
                # The 'phase_in_segment' is crucial for controlling the shape of the foot's path.

                if jtype == 'hip':
                    # Hip: primarily for abduction/adduction (side-to-side balance/spread)
                    # This motion also contributes to forward/backward leg placement.
                    # The phase of this sine wave controls when the leg moves in/out.
                    # Left legs: FL, RL should abduct (positive) then adduct (negative)
                    # Right legs: FR, RR should adduct (negative) then abduct (positive)
                    if leg_prefix in ['FL', 'RL']:
                        # Abduct (out) during swing, slightly adduct during stance
                        angle_offset = hip_abduction_amp * np.sin(2 * np.pi * current_leg_phase) 
                    else:
                        # Adduct (in) during swing, slightly abduct during stance
                        angle_offset = -hip_abduction_amp * np.sin(2 * np.pi * current_leg_phase)

                elif jtype == 'thigh':
                    if is_swing:
                        # Thigh for forward motion (swing) and lift.
                        # Moves forward, then lifts, then comes down and retracts slightly.
                        # A cosine wave provides smooth acceleration/deceleration.
                        # Forward motion: thigh_swing_amp * cos(pi * phase_in_segment)
                        # Lift: stride_vertical_amp * sin(pi * phase_in_segment)
                        
                        # Forward swing (initial part of swing)
                        # The leg moves forward to prepare for contact.
                        # This controls the x-component of the foot trajectory.
                        angle_offset = -thigh_swing_amp * np.cos(np.pi * phase_in_segment) 
                        
                        # Vertical lift during swing (parabolic profile for smooth lift/land)
                        # Ensure this adds to the overall thigh motion.
                        # This term primarily helps lift the foot from the ground.
                        angle_offset += -stride_vertical_amp * np.sin(np.pi * phase_in_segment) * 0.5 # A factor to blend it

                    else: # Stance Phase
                        # Thigh for backward motion (propulsion)
                        # The leg moves backward relative to the body to push the robot forward.
                        angle_offset = thigh_swing_amp * (1 - 2 * phase_in_segment) # Linear or slightly curved backward push

                elif jtype == 'calf':
                    if is_swing:
                        # Calf retracts significantly during swing for ground clearance
                        # It should bend back (positive angle) and then extend (negative angle)
                        angle_offset = calf_swing_retraction * np.sin(np.pi * phase_in_segment)
                    else: # Stance Phase
                        # Calf extends (straightens) during stance for full leg extension and push-off
                        # It should be relatively straight (negative angle) to provide good contact.
                        # A slight "push down" at the end of stance might be beneficial.
                        angle_offset = -calf_stance_extension * (1 - np.cos(np.pi * phase_in_segment)) 
                        # This makes it extend and then perhaps slightly retract towards the end of stance.

                target_angles[jname] = base_angle + angle_offset

            # Apply joint commands
            for jname, target in target_angles.items():
                idx = self.joint_name_to_index[jname]
                p.setJointMotorControl2(self.robot, idx, p.POSITION_CONTROL,
                                        targetPosition=target,
                                        force=250, # Ensure sufficient motor force
                                        maxVelocity=max_joint_velocity)

            p.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step

            # Optional: Print time and global phase to monitor progress
            # print(f"Time: {t:.2f}s, Global Phase: {gait_phase_global:.2f}")

        p.disconnect(self.client)

# === Entry Point ===
if __name__ == "__main__":
    robot_controller = MiniCheetahGaitController(render=True)
    try:
        robot_controller.run(duration_sec=600)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        if p.isConnected():
            p.disconnect()