import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import he_normal
import keras

# Dueling DQN implementation heavily influenced by Abhishek Suran
class DuelingDQN(Model):

    def __init__(self, n_actions):
        super(DuelingDQN, self).__init__()

        # self.model_input = Input(shape=(6,1))
        self.dense1 = Dense(128, activation='relu', kernel_initializer='he_normal')
        self.dense2 = Dense(128, activation='relu', kernel_initializer='he_normal')
        self.val = Dense(1)
        self.adv = Dense(n_actions)

    def call(self, inputs, training=None, mask=None):
        print("DEBUG: inputs shape", inputs.shape)
        # input_layer = self.model_input(inputs)
        # assert(inputs.shape == (6, 1))  # THIS NEEDS TO BE TRUE

        x = self.dense1(inputs)
        x = self.dense2(x)

        v = self.val(x)
        a = self.adv(x)

        q_values = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        print("DEBUG: q_values shape:\t", q_values.shape)

        return q_values

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)

        a = self.adv(x)

        return a

    def get_config(self):
        raise NotImplementedError


"""
https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a
"""
