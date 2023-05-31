import gym
from stable_baselines3 import A2C, PPO

import os
import time
from utils.logger import logger
from snakeenv import SnekEnv

model_fn = PPO
models_dir = f"models/{model_fn.__name__}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TIMESTEPS = 10000

env = SnekEnv()
env.reset()
log = logger(env)
# log.print_obs_action_space()

model = model_fn("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                tb_log_name=f"{model_fn.__name__}")
    # Save the model every 1000 timesteps
    model.save(f"{models_dir}/{TIMESTEPS*i}")


env.close()
