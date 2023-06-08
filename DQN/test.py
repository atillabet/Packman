import os
import argparse

# Set TensorFlow environment variable to suppress warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import gym
import numpy as np
from agent import DQNAgent

# Parse command line arguments
parser = argparse.ArgumentParser(description="DQN Agent for MsPacman")
parser.add_argument("--input-file", type=str, default=None,
                    help="File path to load a pre-trained DQN model (default: None)")
args = parser.parse_args()

# Create the MsPacman environment
env = gym.make('MsPacman-v4')
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Create the DQN agent
agent = DQNAgent(state_shape, action_size)

# Load a trained model
agent.load_model(args.input_file)

num_episodes = 5000

# Start testing episodes
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    counter = 1
    state = np.expand_dims(state, axis=0)

    # Continue the episode until termination
    while not done:
        # render game
        env.render()
        # Choose an action using the DQN agent
        action = agent.get_action(state)
        # Take a step in the environment
        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        total_reward += reward

        if counter % 10 == 0:
            print(" score: ", total_reward, info, action)
        counter += 1

    print("Episode: ", episode, " total reward: ", total_reward)

