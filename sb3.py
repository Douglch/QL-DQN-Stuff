import gym
from stable_baselines3 import A2C, PPO

import os
from utils.logger import logger

models_dir = "models/A2C"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TIMESTEPS = 10000

env = gym.make("LunarLander-v2")
env.reset()
log = logger(env)
# log.print_obs_action_space()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                tb_log_name="A2C")
    # Save the model every 1000 timesteps
    model.save(f"{models_dir}/{TIMESTEPS*i}")

'''
episodes = 10
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
'''

env.close()
