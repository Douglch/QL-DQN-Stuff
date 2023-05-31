from stable_baselines3.common.env_checker import check_env
from snakeenv import SnekEnv

env = SnekEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

episodes = 50

for episodes in range(episodes):
    terminated = False
    obs = env.reset()
    while not terminated:
        random_action = env.action_space.sample()
        print("action:", random_action)
        obs, reward, terminated, info = env.step(random_action)
        print("reward:", reward)

env.close()
