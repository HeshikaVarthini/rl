'''from MinitaurWithSensors_ws import MinitaurWithSensors
from sac_ws import SAC2Agent, train_loop

env = MinitaurWithSensors(render=False)
agent = SAC2Agent(env)
train_loop(env, agent, max_total_steps=200000, max_steps=1000, batch_size=256, intermediate_policies=True, verbose=True)
agent.save_policy("policy1.pth")
'''
import torch
from sac_ws import SAC2Agent,train_loop   # <-- your SAC2Agent code
from MinitaurWithSensors_ws import MinitaurObstacleEnv  # <-- env above

env = MinitaurObstacleEnv(render=False)
agent = SAC2Agent(env)

# Train loop
rewards = train_loop(
    env=env,
    agent=agent,
    max_total_steps=100000,  # adjust for longer training
    max_steps=1000,
    batch_size=256,
    path="./",
    verbose=True
)

# Save trained policy
agent.save_policy("minitaur_obstacle_policy.pth")
env.close()
