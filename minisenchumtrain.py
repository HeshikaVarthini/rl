from minitaur_with_sensors import MinitaurWithSensors
import time

env = MinitaurWithSensors(render="human")
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    time.sleep(1. / 240.)

env.close()
