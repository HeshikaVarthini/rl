from stable_baselines3 import PPO
from quadwalktrain import QuadrupedWalkEnv

env = QuadrupedWalkEnv(render=True)
model = PPO.load("ppo_quadruped_walk")

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
