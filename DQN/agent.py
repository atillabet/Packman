import random
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque


class DQNAgent:
    def __init__(self, state_shape, action_size):
        """
        Initialize the DQNAgent class.

        Parameters:
        - state_shape (tuple): The shape of the state input.
        - action_size (int): The number of possible actions.

        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=50_000)  # Memory buffer to store experience tuples
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 0.0  # Exploration rate
        self.epsilon_decay = 0.999  # Decay rate for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.0001  # Learning rate for the model
        self.model = self._build_model()  # Build the DQN model

    def _build_model(self):
        """
        Build and compile the DQN model.

        Returns:
        - model (Sequential): The compiled DQN model.

        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=self.state_shape, dtype=tf.float32))
        model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=self.state_shape, dtype=tf.float32))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_memory(self, state, action, reward, next_state, done):
        """
        Remember an experience tuple (state, action, reward, next_state, done) in the agent's memory.

        Parameters:
        - state (numpy array): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (numpy array): The next state.
        - done (bool): Whether the episode is done or not.

        """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Get an action to take based on the given state.

        Parameters:
        - state (numpy array): The current state.

        Returns:
        - action (int): The action to take.

        """
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.action_size)  # Exploration: choose random action
            return action

        prob = self.model.predict(state)  # Exploitation: choose action based on the model's prediction
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()

        return int(action.numpy()[0])

    def train(self, batch_size):
        """
        Train the DQN model using experience replay.

        Parameters:
        - batch_size (int): The number of samples to use in each training batch.

        """
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        """
        Save the DQN model to a file.

        Parameters:
        - file_path (str): The path to save the model.

        """
        self.model.save(file_path)

    def load_model(self, file_path):
        """
        Load a saved DQN model from a file.

        Parameters:
        - file_path (str): The path to load the model from.

        """
        self.model = load_model(file_path)
