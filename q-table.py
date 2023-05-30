# Note: Do pip install gym==0.21.0, will not work with the latest version of gym, gymnasium.
import gym
import numpy as np
import matplotlib.pyplot as plt

from utils.logger import logger
from tqdm import tqdm

env = gym.make("MountainCar-v0")

log = logger(env)
# print("_____OBSERVATION SPACE_____ \n")
# print("Observation Space", env.observation_space)
# print("Sample observation", env.observation_space.sample()) # Get a random observation

# print("Observation Space High", env.observation_space.high)
# print("Observation Space Low", env.observation_space.low)

# print("\n _____ACTION SPACE_____ \n")
# print("Action Space Shape", env.action_space.n)
# print("Action Space Sample", env.action_space.sample()) # Take a random action

log.print_observation_space()
log.print_action_space()


# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95  # How important are future actions compared to current actions
EPISODE = 2000  # How many times we are going to run the environment

SHOW_EVERY = 500  # How often we are going to render the environment

EPSILON = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODE // 2
EPSILON_DECAYING_VALUE = EPSILON / \
    (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Create a list of 20s with the length of the observation space
DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OBS_WIN_SIZE = (env.observation_space.high -
                         env.observation_space.low) / DISCRETE_OBS_SIZE

ep_rewards = []
# Aggregate episode rewards for tracking performance
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))
print("q-table shape:", q_table.shape)
print("q-table:", q_table)


def get_discrete_state(state):
    '''
    We need to reference state[0] because env.reset() returns something like "(array([-0.5732655,  0.       ], dtype=float32), {})."
    We only want the array."
    '''
    discrete_state = (state[0] - env.observation_space.low) / \
        DISCRETE_OBS_WIN_SIZE
    return tuple(discrete_state.astype(int))


print("discrete state: ", get_discrete_state(env.reset()))
print("discrete_state type:", type(get_discrete_state(env.reset())))
print(q_table[(0, 1)])
print(np.max(q_table[(0, 1)]))
# for episode in tqdm(range(EPISODE)):
#     episode_reward = 0
#     if episode % SHOW_EVERY == 0:
#         print(episode)
#         render = True
#     else:
#         render = False
#     discrete_state = get_discrete_state(env.reset())
#     done = False
#     while not done:
#         if np.random.random() > EPSILON:
#             action = np.argmax(q_table[discrete_state])
#         else:
#             action = np.random.randint(0, env.action_space.n)
#         new_state, reward, done, _ = env.step(action)
#         episode_reward += reward
#         new_discrete_state = get_discrete_state(new_state)
#         if render:
#             env.render(mode="human")
#         if not done:
#             max_future_q = np.max(q_table[new_discrete_state])
#             current_q = q_table[discrete_state + (action, )]
#             new_q = (1 - LEARNING_RATE) * current_q + \
#                 LEARNING_RATE * (reward + DISCOUNT * max_future_q)
#             q_table[discrete_state + (action, )] = new_q
#         elif new_state[0] >= env.goal_position:
#             print(f"We made it on episode {episode}")
#             q_table[discrete_state + (action, )] = 0

#         discrete_state = new_discrete_state
#     if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
#         EPSILON -= EPSILON_DECAYING_VALUE

#     ep_rewards.append(episode_reward)

#     if not episode % SHOW_EVERY:  # When episode is rendered
#         average_reward = sum(
#             ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
#         aggr_ep_rewards['ep'].append(episode)
#         aggr_ep_rewards['avg'].append(average_reward)
#         aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
#         aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
#         # Current episode, average reward, min reward, max reward
#         print(
#             f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max:{max(ep_rewards[-SHOW_EVERY:])}")
env.close()

# plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
# plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
# plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
# plt.legend(loc=4)  # 4 is the lower right corner
# plt.show()
