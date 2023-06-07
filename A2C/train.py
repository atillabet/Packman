import os
import numpy as np
import gym
from agent import A2CAgent
import tensorflow as tf

# Set TensorFlow log level to suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

# Set the number of episodes for training
num_episodes = 1000

# Create a dictionary to store the training history
training_history = {'episode': [], 'total_reward': []}

# Start training loop
for episode in range(num_episodes):
    # Reset the environment for a new episode
    state = env.reset()
    done = False
    total_reward = 0
    rewardsss = 0
    prev_lives = 3
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

        # Update the accumulated reward based on the reward received
        curr_lives = info['ale.lives']
        if reward == 10:
            rewardsss += 1
            dots += 1
        elif reward == 50:
            rewardsss += 2
        elif reward >= 200:
            rewardsss += 4
        if curr_lives < prev_lives:
            rewardsss -= 8
        if dots == 10:
            dots = 0
            rewardsss += 5

        # Update the accumulated reward based on the current and next Pacman positions
        pacman_position_current = np.argwhere(state == 210)[0]
        pacman_position_next = np.argwhere(next_state == 210)[0]
        if (pacman_position_current == pacman_position_next).all() and action != 0:
            rewardsss -= 20

        # Train the agent with the current transition
        agent.train(state, action, rewardsss, next_state, done)

        # Update the current state
        state = next_state

        # Update the total reward
        total_reward += reward

        # Print the score every 10 steps
        if counter % 10 == 0:
            print(" score: ", total_reward, info, action)

        counter += 1

    # Print the episode number and the total reward
    print("Episode: ", episode, " total reward: ", total_reward)

    # Save the model and training history every 50 episodes
    if episode % 50 == 0:
        agent.save_model()
        episodes = np.array(training_history['episode'])
        rewards = np.array(training_history['total_reward'])
        np.savez('training_history.npz', episodes=episodes, rewards=rewards)

# Save the final model and training history
agent.save_model()
episodes = np.array(training_history['episode'])
rewards = np.array(training_history['total_reward'])
np.savez('training_history.npz', episodes=episodes, rewards=rewards)

# Close the environment
env.close()
