import gym
from stable_baselines3 import A2C, PPO

import os
from utils.logger import logger

# Edit the model_fn to the algorithm you want to use
model_fn = PPO
env = gym.make("LunarLander-v2")
env.reset()
log = logger(env)
# log.print_obs_action_space()

models_dir = f"models/{model_fn.__name__}"
# Edit the number (timestep) to the model you want to load
model_path = f"{models_dir}/290000"

model = model_fn.load(model_path, env=env)

TIMESTEPS = 10000


episodes = 10
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
