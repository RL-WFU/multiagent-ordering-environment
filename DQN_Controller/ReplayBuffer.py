import numpy as np
import os
from datetime import datetime


class ReplayBuffer:
    """
    Replay Buffer -- Stores and retrieves experience
    Inspired and adapted from Sebastian Theiler's code
    """

    def __init__(self, map_height, map_width, directory_path: str, batch_size=32, size=10000, minimum_buffer=200,
                 use_per=False):
        # Replay buffer
        self.size = size
        self.minimum_buffer = minimum_buffer  # FIXME: might not be necessary
        self.batch_size = batch_size
        self.use_per = use_per

        # Input-related
        self.height = map_height
        self.width = map_width

        # Pre-allocate memory
        self.states = np.empty((self.size, self.height, self.width, 1), dtype=np.uint8)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.next_states = np.empty((self.size, self.height, self.width, 1), dtype=np.uint8)
        self.dones = np.empty(self.size, dtype=np.bool)
        if use_per:
            self.priorities = np.zeros(self.size, dtype=np.float32)

        # Tracker
        self.count = 0
        self.pointer = 0

        # Time
        self.directory_path = directory_path

    def add_experience(self, state, action, reward, next_state, done, reward_clipping=True):
        """
        Store one transition

        :param reward_clipping:
        :param state: original state
        :param action: action taken
        :param reward: reward from taking action at original state
        :param next_state: state after taking action
        :param done: indicates whether the episode terminated after taking the action
        :return: None
        """

        # FIXME: Consider processing state and next_state here... and check that they are processed somewhere

        """
        I considered making this a deque object like in other DQN implementations.
        One could test which is more efficient, but I like this pre-allocation method 
        and wanted to play around with it.        
        """
        if reward_clipping:
            reward = np.sign(reward)

        # Write experience in pre-allocated arrays
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done

        # Update tracking
        self.count = max(self.count, self.pointer + 1)  # Acts as a pseudo len() function
        self.pointer = (self.pointer + 1) % self.size  # Starts rewriting when the replay buffer is full

    def sample_experience(self, priority_scale=0.0):
        """
        Returns a minibatch from experience
        :param priority_scale:
        :return: a tuple {state, action, reward, next_state} sampled from memory
        """

        # Ensure that there are enough observations to get a batch
        if self.count < self.batch_size:
            raise ValueError('The batch_size exceeds the number of observations saved.')

        # Get sampling probabilities from priority list
        if self.use_per:
            scaled_priorities = self.priorities[self.count - 1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of indices (without replacement)
        indices = np.random.choice(np.arange(self.count), self.batch_size, replace=False)

        return self.states[indices], self.actions[indices], self.rewards[indices], \
               self.next_states[indices], self.dones[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
            :param offset:
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self):
        if not os.path.isdir(self.directory_path + '/ReplayBuffer'):
            os.mkdir(self.directory_path + '/ReplayBuffer')

        np.save(self.directory_path + 'ReplayBuffer/states.npy', self.states)
        np.save(self.directory_path + 'ReplayBuffer/actions.npy', self.actions)
        np.save(self.directory_path + 'ReplayBuffer/rewards.npy', self.rewards)
        np.save(self.directory_path + 'ReplayBuffer/next_states.npy', self.next_states)
        np.save(self.directory_path + 'ReplayBuffer/dones.npy', self.dones)

    def load(self):
        self.actions = np.load(self.directory_path + 'actions.npy')

        np.load(self.directory_path + 'ReplayBuffer/states.npy')
        np.load(self.directory_path + 'ReplayBuffer/actions.npy')
        np.load(self.directory_path + 'ReplayBuffer/rewards.npy')
        np.load(self.directory_path + 'ReplayBuffer/next_states.npy')
        np.load(self.directory_path + 'ReplayBuffer/dones.npy')


"""
Useful Articles:
Building a DQN in TensorFlow 2.0 -- https://bit.ly/3fqrp95 (Shortened link, should lead to Medium.com)
Implementing Prioritized Experience Replay -- https://arxiv.org/pdf/1511.05952.pdf
"""
