from datetime import datetime
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

from DQN_Controller.Architecture import DuelingDQN
from DQN_Controller.ReplayBuffer import ReplayBuffer


class DQNAgent:

    def __init__(self, states_shape, actions_size, batch_size=32, minimum_buffer=1000, maximum_buffer=100000):

        # Save/Load parameters
        # FIXME: Once implementation is stable, move this to the train.py file for simplicity
        now = datetime.now()
        self.directory_path = now.strftime('%b%d/%H%M')

        # Action and state information
        self.states_shape = states_shape  # Shape of states
        self.actions_size = actions_size  # Action space size
        self.env_height = states_shape[0]
        self.env_width = states_shape[1]

        # Replay Buffer parameters
        self.batch_size = batch_size
        self.minimum_buffer = minimum_buffer
        self.maximum_buffer = maximum_buffer
        self.use_per = False  # NOT IMPLEMENTED
        self.replay = ReplayBuffer(self.env_height, self.env_width, self.directory_path, size=maximum_buffer)

        # Parameters
        self.gamma = 0.95  # Discount Rate
        self.learning_rate = 0.0001
        self.epsilon = 1.0  # Exploration Rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Build Q Networks
        self.main_network = self.build_dueling()
        self.target_network = self.build_dueling()

        self.update_frequency = 200
        self.training_step = 0

    def build_model(self):
        opt = Adam(self.learning_rate)

        model = DuelingDQN(self.actions_size)
        model.compile(loss=Huber(), optimizer=opt)

        return model

    def build_dueling(self):

        model_input = Input(shape=(self.env_height, self.env_width))
        dense1 = Dense(128, activation='relu', kernel_initializer='he_normal')(model_input)
        dense2 = Dense(128, activation='relu', kernel_initializer='he_normal')(dense1)
        dense2 = Flatten()(dense2)

        val = Dense(1, activation=None)(dense2)
        adv = Dense(self.actions_size, activation=None)(dense2)

        q = val + (adv - tf.math.reduce_mean(adv, axis=1, keepdims=True))

        model = Model(model_input, q)
        model.compile(Adam(self.learning_rate), loss=Huber())

        def advantage(state):
            x = dense1(state)
            x = dense2(x)
            a = adv(x)

            return a

        return model

    # Take an action in the environment
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions_size)

        else:
            """
            actions = self.main_network.advantage(np.array([state]))  # FIXME: predict vs advantage here?
            return np.argmax(actions)
            """
            # print("DEBUG ACT:", state.shape)
            state = state.reshape((-1, self.env_height, self.env_width))
            # print("DEBUG STATE SHAPE:", state.shape)
            q_vals = self.main_network.predict(state)[0]
            # print("q_vals assigned")
            return q_vals.argmax()

    def train(self):
        """
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # Target DQN estimates the values of next states, but action is picked from Main DQN
            next_states_val = self.target_network.predict(next_states)
            print('DEBUG: ', self.main_network.predict(next_states))
            action_from_main = np.argmax(self.main_network.predict(next_states), axis=2)
            # action_from_main = np.argmax(action_from_main, axis=1)

            # Look at targets and train on batch
            q_target = self.main_network.predict(states)  # Make copy if you want to leave target intact?
            q_target[batch_index, None, actions] = rewards + self.gamma * next_states_val[
                batch_index, None, action_from_main]  # removed dones, add back in?
            self.main_network.train_on_batch(states, q_target)

            # Update epsilon and timestep here
            self.update_epsilon()
            self.training_step += 1

            # Update target every self.update_frequency steps
            if self.training_step % self.update_frequency == 0:
                self.update_target()
        """

        if self.replay.count < self.batch_size:
            raise ValueError('Replay buffer not yet filled...'
                             '\nYou should not have entered this function')

        states, actions, rewards, next_states, dones = self.replay.sample_experience()

        arg_q_max = self.main_network.predict(next_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_network.predict(next_states)
        double_q = future_q_vals[range(self.batch_size), arg_q_max]

        # Calculate targets
        target_q = rewards + self.gamma * double_q * (1 - dones)

        # Use targets to calculate loss
        with tf.GradientTape() as tape:
            q_values = self.main_network(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.actions_size,
                                                            dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

        model_gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.main_network.optimizer.apply_gradients(zip(model_gradients, self.main_network.trainable_variables))

        # Update epsilon and timestep here
        self.update_epsilon()
        self.training_step += 1

        # Update target every self.update_frequency steps
        if self.training_step % self.update_frequency == 0:
            self.update_target()

        return float(loss.numpy()), error

    def update_target(self):
        self.target_network.set_weights(self.main_network.get_weights())

    # Wrapper for update replay buffer
    def update_memory(self, state, action, reward, next_state, done):
        self.replay.add_experience(state, action, reward, next_state, done)

    def update_epsilon(self):
        # FIXME: remove this error after debug
        if self.replay.count < self.minimum_buffer:
            raise ValueError('DEBUG: You should not be calling this function this early')

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, **kwargs):
        if not os.path.isdir(self.directory_path):
            os.mkdir(self.directory_path)

        # Save networks
        self.main_network.save(self.directory_path + '/Networks/dqn.h5')
        self.target_network.save(self.directory_path + '/Networks/target_dqn.h5')

        # Save replay buffer
        self.replay.save()

        # Save metadata
        with open(self.directory_path + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'replay_count': self.replay.count, 'replay_pointer': self.replay.pointer},
                                **kwargs}))  # save replay_buffer information and any other information

    def load(self):
        raise NotImplementedError
