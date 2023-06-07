# Import necessary libraries
import os
import numpy as np
import gym
from agent import A2CAgent
import tensorflow as tf
import argparse

# Set TensorFlow log level to suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parse command line arguments
parser = argparse.ArgumentParser(description="A2C Agent for MsPacman")
parser.add_argument("--input-file", type=str, default=None,
                    help="File path to load a pre-trained A2C model (default: None)")
args = parser.parse_args()

# Configure GPU memory growth to avoid memory errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Create the environment
env = gym.make('MsPacman-v4')

# Get the shape of the state and the number of possible actions
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Create an A2C agent
agent = A2CAgent(state_shape, action_size)

# Load the pre-trained model
agent.load_model()

# Set the number of episodes for testing
num_episodes = 1000

# Create a dictionary to store the testing history
test_history = {'episode': [], 'total_reward': []}

# Start testing loop
for episode in range(num_episodes):
    # Reset the environment for a new episode
    state = env.reset()
    done = False
    total_reward = 0
    counter = 1
    dots = 0

    # Run the episode until termination
    while not done:
        # Render the environment
        env.render(mode="human")

        # Convert the state to a numpy array
        state = np.array(state)

        # Get the action to take from the agent
        action = agent.get_action(state)

        # Perform the action and observe the next state, reward, termination, and info
        next_state, reward, done, info = env.step(action)

        # Update the total reward
        total_reward += reward

        # Print the score every 10 steps
        if counter % 10 == 0:
            print(" score: ", total_reward, info, action)

        counter += 1

    # Print the episode number and the total reward
    print("Episode: ", episode, " total reward: ", total_reward)

# Close the environment
env.close()
