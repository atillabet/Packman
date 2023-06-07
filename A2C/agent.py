import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model

class Actor(tf.keras.Model):
    def __init__(self, state_shape):
        super().__init__()
        # Define the layers of the actor model
        self.conv1 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.conv2 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.flatten = Flatten()
        self.dense1 = Dense(24, activation='relu')
        self.dense2 = Dense(9, activation='relu')

    def call(self, x):
        # Define the forward pass of the actor model
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
        # Define the layers of the critic model
        self.conv1 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.conv2 = Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=state_shape, dtype=tf.float32)
        self.flatten = Flatten()
        self.dense1 = Dense(24, activation='relu')
        self.dense2 = Dense(1, activation='relu')

    def call(self, x):
        # Define the forward pass of the critic model
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class A2CAgent:
    def __init__(self, state_shape, action_size, actor_lr=0.0001, critic_lr=0.0001, gamma=0.90, epsilon=0.2, max_consecutive_actions=7, penalty_factor=0):
        """
        Initialize the A2CAgent class.

        Parameters:
        - state_shape (tuple): The shape of the state input.
        - action_size (int): The number of possible actions.
        - actor_lr (float): The learning rate for the actor model.
        - critic_lr (float): The learning rate for the critic model.
        - gamma (float): The discount factor for future rewards.
        - epsilon (float): The exploration rate.
        - max_consecutive_actions (int): The maximum number of consecutive actions before applying a penalty.
        - penalty_factor (float): The penalty factor for consecutive actions.

        """
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

    def _actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * td
        return loss

    def train(self, state, action, reward, next_state, done):
        """
        Perform a single training step on the actor and critic models.

        Parameters:
        - state (array): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (array): The next state.
        - done (bool): Whether the episode is done or not.

        """
        state = np.array([state])
        next_state = np.array([next_state])
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(state, training=True)
            v = self.critic(state, training=True)
            vn = self.critic(next_state, training=True)
            td = reward + self.gamma * vn * (1 - int(done)) - v
            a_loss = self._actor_loss(p, action, td)
            c_loss = td ** 2
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

    def save_model(self, actor_path, critic_path):
        """
        Save the actor and critic models to files.

        """
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load_model(self, actor_path, critic_path):
        """
        Load saved actor and critic models from files.

        """
        self.actor = load_model(actor_path)
        self.critic = load_model(critic_path)
