import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
env.reset()
# print("_____OBSERVATION SPACE_____ \n")
# print("Observation Space", env.observation_space)
# print("Sample observation", env.observation_space.sample()) # Get a random observation

# print("Observation Space High", env.observation_space.high)
# print("Observation Space Low", env.observation_space.low)

# print("\n _____ACTION SPACE_____ \n")
# print("Action Space Shape", env.action_space.n)
# print("Action Space Sample", env.action_space.sample()) # Take a random action

discrete_obs_size = [20] * len(env.observation_space.high) # Create a list of 20s with the length of the observation space
# print("discrete_os_size:", discrete_obs_size)
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / discrete_obs_size # 
# print("discrete_obs_win_size:", discrete_obs_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(discrete_obs_size + [env.action_space.n]))
print(q_table.shape)

done = False

while not done:
    action = 2
    new_state, reward, done, _, _ = env.step(action)
    
env.close()