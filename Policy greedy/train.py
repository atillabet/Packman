import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import gym
from agent import PolicyGradientAgent
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

env = gym.make('MsPacman-v4')
state_shape = env.observation_space.shape
action_size = env.action_space.n
agent = PolicyGradientAgent(state_shape, action_size)

num_episodes = 1000
batch_size = 32
training_history = {'episode': [], 'total_reward': []}
# agent.load_model()

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    counter = 1
    rewardsss = 0
    prev_lives = 3
    dots = 0
    while not done:
        env.render()
        state = np.array(state)

        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        curr_lives = info['ale.lives']

        if reward == 10:
            rewardsss += 1
            dots += 1
        elif reward == 50:
            rewardsss += 2
        elif reward >= 200:
            rewardsss += 3
        if curr_lives < prev_lives:
            rewardsss -= 8
        if dots == 10:
            dots = 0
            rewardsss += 6

        pacman_position_current = np.argwhere(state == 210)[0]
        pacman_position_next = np.argwhere(next_state == 210)[0]
        if (pacman_position_current == pacman_position_next).all() and action != 0:
            rewardsss -= 20
        agent.store_transition(state, action, reward)

        agent.train(state, action, rewardsss, next_state, done)

        state = next_state
        total_reward += reward

        if counter % 10 == 0:
            print(" score: ", total_reward, info, action)
        counter += 1

    print("Episode: ", episode, " score: ", total_reward)
    training_history['episode'].append(episode)
    training_history['total_reward'].append(total_reward)

    if episode % 50 == 0:
        print("--------SAVING MODEL--------")
        agent.save_model()
        episodes_ = np.array(training_history['episode'])
        rewards_ = np.array(training_history['total_reward'])

        np.savez('training_history.npz', episodes=episodes_, rewards=rewards_)


agent.save_model()
episodes = np.array(training_history['episode'])
rewards = np.array(training_history['total_reward'])

np.savez('training_history.npz', episodes=episodes, rewards=rewards)