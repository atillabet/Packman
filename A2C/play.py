import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import gym
from agent import A2CAgent
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

env = gym.make('MsPacman-v4')
state_shape = env.observation_space.shape
print(state_shape)
action_size = env.action_space.n
agent = A2CAgent(state_shape, action_size)
# agent.load_model()

num_episodes = 1000
training_history = {'episode': [], 'total_reward': []}
for episode in range(num_episodes):
    state = env.reset()
    # env.render()
    done = False
    total_reward = 0
    rewardsss = 0
    prev_lives = 3
    counter = 1
    dots = 0
    while not done:
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
            rewardsss += 5
        if curr_lives < prev_lives:
            rewardsss -= 3
        if dots == 10:
            dots = 0
            rewardsss += 4
        agent.train(state, action, rewardsss, next_state, done)

        state = next_state

        total_reward += reward
        if counter % 10 == 0:
            print(" score: ", total_reward, info, action)
        counter += 1

    print("Episode: ", episode, " total reward: ", total_reward)

    if episode % 50 == 0:
        agent.save_model()
        episodes = np.array(training_history['episode'])
        rewards = np.array(training_history['total_reward'])

        np.savez('training_history.npz', episodes=episodes, rewards=rewards)

agent.save_model()
episodes = np.array(training_history['episode'])
rewards = np.array(training_history['total_reward'])

np.savez('training_history.npz', episodes=episodes, rewards=rewards)
