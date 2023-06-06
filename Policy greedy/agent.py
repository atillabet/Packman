import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


class PolicyGradientAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.01, discount_factor=0.99, batch_size=32, epsilon=0.0):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.optimizer = Adam(lr=learning_rate)
        self.model = self.build_model()

        self.states = []
        self.actions = []
        self.rewards = []

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=self.state_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def get_action(self, state):
        action_probs = self.model.predict(state)[0]
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(action_probs)

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def discount_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards)
        cumulative_reward = 0
        for t in reversed(range(len(self.rewards))):
            cumulative_reward = cumulative_reward * self.discount_factor + self.rewards[t]
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards

    def train(self):
        num_batches = len(self.states) // self.batch_size
        indices = np.arange(len(self.states))

        for _ in range(num_batches):
            batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
            batch_states = np.vstack([self.states[i] for i in batch_indices])
            batch_actions = np.array([self.actions[i] for i in batch_indices])
            batch_discounted_rewards = self.discount_rewards()[batch_indices]

            action_masks = tf.one_hot(batch_actions, self.action_size)

            with tf.GradientTape() as tape:
                logits = self.model(batch_states, training=True)
                loss = tf.reduce_mean(-tf.math.log(tf.reduce_sum(action_masks * logits, axis=1)) * batch_discounted_rewards)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.states = []
        self.actions = []
        self.rewards = []


    def save_model(self):
        self.model.save("A2CModel.h")

    def load_model(self):
        self.model = load_model("A2CModel.h")