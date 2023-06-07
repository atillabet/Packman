import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gym
import numpy as np
from agent import DQNAgent
import tensorflow as tf
from collections import deque

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

env = gym.make('MsPacman-v4')
state_shape = env.observation_space.shape
action_size = env.action_space.n

agent = DQNAgent(state_shape, action_size)
agent.load_model()

batch_size = 32
num_episodes = 5000
moves = deque(maxlen=6)

for i in range(len(moves)):
    moves[i] = -1

moves.append(-2)

def are_all_elements_same(deque_):
    lst = list(deque_)

    return all(elem == lst[0] for elem in lst)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    rewardsss = 0
    prev_lives = 3
    counter = 1
    state = np.expand_dims(state, axis=0)
    while not done:
        # env.render()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        curr_lives = info['ale.lives']
        if reward == 10:
            rewardsss += 4
        elif reward == 50:
            rewardsss += 2
        elif reward >= 200:
            rewardsss += 4
        if curr_lives < prev_lives:
            rewardsss -= 8

        moves.append(action)
        if are_all_elements_same(moves):
            rewardsss -= 10

        pacman_position_current = np.argwhere(state == 210)[0]
        pacman_position_next = np.argwhere(next_state == 210)[0]
        if (pacman_position_current == pacman_position_next).all() and action != 0:
            rewardsss -= 20

        agent.remember(state, action, rewardsss, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if counter % 10 == 0:
            print(" score: ", total_reward, info, action)
        counter += 1

    print("Episode: ", episode, " total reward: ", total_reward)

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if episode % 10 == 0:
        agent.save_model()