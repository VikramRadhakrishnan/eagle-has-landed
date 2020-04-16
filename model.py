import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization, Activation
from tensorflow.keras import Model

tf.keras.backend.set_floatx('float64')

# Actor model defined using Keras

class Q_Table:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, name="Q_Table"):
        """Initialize parameters.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Dimensions of 1st hidden layer
            fc2_units (int): Dimensions of 2nd hidden layer
            name (string): Name of the model
        """
        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.name = name
        
        # Build the actor model
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = Input(shape=self.state_size)
        
        # Add hidden layers
        net = Dense(units=self.fc1_units, activation='relu', kernel_initializer='glorot_uniform')(states)
        net = Dense(units=self.fc2_units, activation='relu', kernel_initializer='glorot_uniform')(net)

        # Add final output layer with linear activation
        Q_values = Dense(units=self.action_size, activation='linear', kernel_initializer='glorot_uniform')(net)

        # Create Keras model
        self.model = Model(inputs=states, outputs=Q_values, name=self.name)
