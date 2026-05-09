from stable_baselines3 import PPO
from MiniCheetahEnv import MiniCheetahEnv  # Adjust the import path if needed

# Load environment and model
env = MiniCheetahEnv(render=True)
model = PPO.load("ppo_mini_cheetah")

obs = env.reset()
for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()

# Create environment with GUI
'''env = MiniCheetahEnv(render=True)

# Load trained model
model = PPO.load("ppo_minicheetah")

obs = env.reset()
done = False

while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    time.sleep(1.0 / 60.0)  # Optional: slows simulation

    if done:
        obs = env.reset()
'''