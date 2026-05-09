import os
import time
from stable_baselines3 import PPO
from minitaur_obstacle_env import MinitaurObstacleEnv

def main():
    # Load environment with rendering
    env = MinitaurObstacleEnv(render=True)

    # Load trained PPO model
    model_path = "./models_minitaur_obstacle/ppo_minitaur_final.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    model = PPO.load(model_path, env=env)

    # Evaluate over 5 episodes
    for ep in range(5):
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            env.render()
            time.sleep(0.01)  # slow down rendering

        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
