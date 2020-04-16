from model import Q_Table
from replay_buffer import ReplayBuffer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

import numpy as np
import random

# Deep Q Network Agent
class DQN():
    """Reinforcement Learning agent that learns using DQN."""
    def __init__(self, state_size, action_size, lr,
                 random_seed, buffer_size, batch_size,
                 gamma, tau, n_time_steps, n_learn_updates, device):

        self.state_size = state_size
        self.action_size = action_size
        
        self.lr = lr

        # Q Network (w/ Target Network)
        self.q_local = Q_Table(state_size, action_size, name="Q_local")
        self.q_target = Q_Table(state_size, action_size, name="Q_target")
        self.q_optimizer = Adam(learning_rate=self.lr)
        
        # Initialize target model parameters with local model parameters
        self.q_target.model.set_weights(self.q_local.model.get_weights())

        # Replay memory
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, random_seed)

        # Algorithm parameters
        self.gamma = gamma                     # discount factor
        self.tau = tau                         # for soft update of target parameters
        self.n_time_steps = n_time_steps       # number of time steps before updating network parameters
        self.n_learn_updates = n_learn_updates # number of updates per learning step

        # Device
        self.device = device

    def step(self, time_step, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        if time_step % self.n_time_steps != 0:
            return

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            
            # Train the network for a number of epochs specified by the parameter
            for i in range(self.n_learn_updates):
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma)

    def act(self, state, epsilon=0):
        """Returns actions for given state as per current policy."""
        state = np.expand_dims(state, axis=0)
        q_values = self._act_tf(tf.constant(state))
        q_values = q_values.numpy()[0]

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(q_values)
        else:
            return random.choice(np.arange(self.action_size))
    
    @tf.function
    def _act_tf(self, state):
        """Returns actions for given state as per epsilon greedy policy."""
        q_values = self.q_local.model(state)
        return q_values

    def learn(self, experiences, gamma):
        """Update Q parameters using given batch of experience tuples.
        """
        self._learn_tf(experiences, tf.constant(self.gamma, dtype=tf.float64)) 
    
    @tf.function
    def _learn_tf(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        with tf.GradientTape() as tape:
            # Get max predicted Q values from target models
            Q_targets_next = tf.reduce_max(self.q_target.model(next_states), axis=-1)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Get expected Q values from local model
            Q_expected = self.q_local.model(states)
            idx = tf.cast(actions, tf.int32)
            Q_expected = tf.gather_nd(Q_expected, tf.stack([tf.range(Q_expected.shape[0]), idx], axis=1))
            # Calculate the loss
            loss = MSE(Q_targets, Q_expected)
        
        # Minimize the loss
        grad = tape.gradient(loss, self.q_local.model.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grad, self.q_local.model.trainable_variables))

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.q_local.model, self.q_target.model, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: TF2 model
            target_model: TF2 model
            tau (float): interpolation parameter 
        """
        for target_var, local_var in zip(target_model.weights, local_model.weights):
            target_var.assign(tau * local_var + (1.0 - tau) * target_var)
