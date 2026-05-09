import numpy as np
import tensorflow as tf
from ddpg import DDPGAgent
from quadruped_env import QuadrupedPyBulletEnv

def train_quadruped():
    env = QuadrupedPyBulletEnv(render_mode='human')  # Updated to match the correct keyword
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = DDPGAgent(obs_dim, act_dim, act_limit)

    num_episodes = 500
    max_steps = 1000
    batch_size = 64
    start_steps = 1000
    update_after = 1000
    update_every = 50

    total_steps = num_episodes * max_steps
    obs, _ = env.reset()
    episode_return, episode_length = 0, 0

    for t in range(total_steps):
        if t > start_steps:
            act = agent.get_action(obs)
        else:
            act = env.action_space.sample()

        next_obs, reward, done, truncated, _ = env.step(act)
        agent.buffer.store(obs, act, reward, next_obs, done)
        obs = next_obs
        episode_return += reward
        episode_length += 1

        if done or truncated or (episode_length == max_steps):
            print(f"Episode Return: {episode_return:.2f}, Length: {episode_length}")
            obs, _ = env.reset()
            episode_return, episode_length = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                agent.update(batch_size)

    env.close()
    agent.save('ddpg_quadruped.h5')

if __name__ == '__main__':
    train_quadruped()
