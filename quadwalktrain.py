 # <-- folder only
from stable_baselines3 import PPO
from quad_env import QuadrupedWalkEnv
'''
if __name__ == "__main__":
    env = QuadrupedWalkEnv(render=False)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_quadruped_walk")
    env.close()'''

'''from stable_baselines3.common.vec_env import DummyVecEnv
from quad_env import QuadrupedWalkEnv
from stable_baselines3 import PPO
env = DummyVecEnv([lambda: QuadrupedWalkEnv(render=False)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_quadruped_walk")
env.close()'''
env = QuadrupedWalkEnv(render=True)
observation = env.reset()
import time

# Add sliders for each controllable joint
slider_ids = []
for idx, joint_index in enumerate(env.joint_indices):
    joint_name = env.joint_names[idx]
    # Set slider at current joint value, range -2 to 2 radians (adjust range if needed)
    cur_val = p.getJointState(env.robot, joint_index)[0]
    slider_id = p.addUserDebugParameter(joint_name, -2.0, 2.0, cur_val)
    slider_ids.append(slider_id)

# Let user move sliders and update robot in realtime
print("Use the sliders in the PyBullet GUI to manually pose the robot.")
print("Watch the robot's stability in the window.")
print("When the robot stands as desired, note the slider values for each joint.")
