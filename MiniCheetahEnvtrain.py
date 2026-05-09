from stable_baselines3 import PPO
from MiniCheetahEnv import MiniCheetahEnv
from stable_baselines3.common.vec_env import DummyVecEnv
'''
env = DummyVecEnv([lambda: MiniCheetahEnv(render=False)])

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cheetah_log/")
model.learn(total_timesteps=2_000_000)  # 🚨 Long training!

model.save("ppo_minicheetah")
env.close()
'''
'''
from stable_baselines3 import PPO

if __name__ == "__main__":
    from MiniCheetahEnv import MiniCheetahEnv   # or paste above class in this file

    env = MiniCheetahEnv(render=False)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cheetah_tensorboard/")

    model.learn(total_timesteps=200_000)

    # Save and test
    model.save("ppo_mini_cheetah")
    env = MiniCheetahEnv(render=True)
    obs = env.reset()
    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()'''
env = MiniCheetahEnv(render=True)
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # or np.zeros(12) for testing
    obs, reward, done, _ = env.step(action)
    if done:
        print("Robot fell. Resetting...")
        obs = env.reset()

