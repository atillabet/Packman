import os
import argparse

# Set TensorFlow environment variable to suppress warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gym
import numpy as np
from agent import DQNAgent
import tensorflow as tf
from collections import deque

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parse command line arguments
parser = argparse.ArgumentParser(description="DQN Agent for MsPacman")
parser.add_argument("--output-file", type=str, default=None,
                    help="File path to save the trained DQN model (default: None)")
parser.add_argument("--input-file", type=str, default=None,
                    help="File path to load a pre-trained DQN model (default: None)")
args = parser.parse_args()

# Create the MsPacman environment
env = gym.make('MsPacman-v4')
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Create the DQN agent
agent = DQNAgent(state_shape, action_size)

# Load a pre-trained model if specified
if args.input_file:
    agent.load_model(args.input_file)

batch_size = 32
num_episodes = 5000
moves = deque(maxlen=6)

# Initialize the moves deque with -1
for i in range(len(moves)):
    moves[i] = -1

moves.append(-2)

# Start training episodes
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    rewardsss = 0
    prev_lives = 3
    counter = 1
    state = np.expand_dims(state, axis=0)

    # Continue the episode until termination
    while not done:
        # Choose an action using the DQN agent
        action = agent.get_action(state)
        # Take a step in the environment
        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        # Calculate modified rewards
        curr_lives = info['ale.lives']
        if reward == 10:
            rewardsss += 4
        elif reward == 50:
            rewardsss += 2
        elif reward >= 200:
            rewardsss += 4
        if curr_lives < prev_lives:
            rewardsss -= 8

        # Append the current action to the moves deque
        moves.append(action)

        # Penalize if the Pacman position remains the same and the action is not no-op
        pacman_position_current = np.argwhere(state == 210)[0]
        pacman_position_next = np.argwhere(next_state == 210)[0]
        if (pacman_position_current == pacman_position_next).all() and action != 0:
            rewardsss -= 20

        # Store the transition in the agent's memory
        agent.update_memory(state, action, rewardsss, next_state, done)
        state = next_state
        total_reward += reward

        if counter % 10 == 0:
            print(" score: ", total_reward, info, action)
        counter += 1

    print("Episode: ", episode, " total reward: ", total_reward)

    # Perform a replay update at the end of the episode if memory size is sufficient
    if len(agent.memory) > batch_size:
        agent.train(batch_size)

    # Save the model every 10 episodes
    if episode % 10 == 0:
        agent.save_model(args.output_file)
