import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model

class Actor(tf.keras.Model):
    def __init__(self, state_shape):
        super().__init__()
        self.conv1 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.conv2 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.flatten = Flatten()
        self.dense1 = Dense(24, activation='relu')
        self.dense2 = Dense(9, activation='relu')

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class Critic(tf.keras.Model):
    def __init__(self, state_shape):
        super().__init__()
        self.conv1 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.conv2 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.flatten = Flatten()
        self.dense1 = Dense(24, activation='relu')
        self.dense2 = Dense(1, activation='relu')

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class A2CAgent:
    def __init__(self, state_shape, action_size, actor_lr=0.0001, critic_lr=0.0001, gamma=0.90, epsilon=0.2, max_consecutive_actions=7, penalty_factor=0):
        self.state_shape = state_shape
        self.action_size = action_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_consecutive_actions = max_consecutive_actions
        self.penalty_factor = penalty_factor

        self.actor = Actor(self.state_shape)
        self.critic = Critic(self.state_shape)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.consecutive_actions = []

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
             action = np.random.randint(self.action_size)
             return action

        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()

        return int(action.numpy()[0])

    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * td
        return loss

    def update_consecutive_actions(self, action):
        self.consecutive_actions.append(action)
        if len(self.consecutive_actions) > self.max_consecutive_actions:
            self.consecutive_actions.pop(0)

    def check_same_actions(self):
        return len(set(self.consecutive_actions)) == 1

    def apply_penalty(self, reward):
        return reward - self.penalty_factor

    def train(self, state, action, reward, next_state, done):

        self.update_consecutive_actions(action)

        if self.check_same_actions():
            reward = self.apply_penalty(reward)

        self.consecutive_actions[-1] = action

        state = np.array([state])
        next_state = np.array([next_state])
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(state, training=True)
            v = self.critic(state, training=True)
            vn = self.critic(next_state, training=True)
            td = reward + self.gamma * vn * (1 - int(done)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td ** 2
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

    def save_model(self):
        self.actor.save("A2CActor4.h")
        self.critic.save("A2CCritic4.h")
        optimizer_weights = self.actor_optimizer.get_weights()
        np.savez('actor_weights.npz', optimizer_weights=optimizer_weights)
        optimizer_weights = self.critic_optimizer.get_weights()
        np.savez('critic_weights.npz', optimizer_weights=optimizer_weights)

    def load_model(self):
        self.actor = load_model("A2CActor3.h")
        self.critic = load_model("A2CCritic3h")
        loaded_weights = np.load('actor_weights.npz', allow_pickle=True)
        optimizer_weights = loaded_weights['optimizer_weights'].astype(object)
        # self.actor_optimizer.set_weights(optimizer_weights)
        loaded_weights = np.load('critic_weights.npz', allow_pickle=True)
        optimizer_weights = loaded_weights['optimizer_weights'].astype(object)
        # self.critic_optimizer.set_weights(optimizer_weights)
