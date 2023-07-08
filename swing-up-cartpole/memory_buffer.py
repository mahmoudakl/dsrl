import numpy as np


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, store_spikes=False, simtime=10):
        self.store_spikes = store_spikes

        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        if self.store_spikes:
            self.state_spikes_memory = np.zeros((self.mem_size, input_shape[0], simtime))
            self.new_state_spikes_memory = np.zeros((self.mem_size, input_shape[0], simtime))

    def store_transition(self, state, action, reward, state_, done, state_spikes=None,
                         state_spikes_=None):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        if self.store_spikes:
            self.state_spikes_memory[index] = state_spikes
            self.new_state_spikes_memory[index] = state_spikes_

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        if self.store_spikes:
            states_spikes = self.state_spikes_memory[batch]
            states_spikes_ = self.new_state_spikes_memory[batch]
            return states, states_spikes, actions, rewards, states_, states_spikes_, dones

        return states, actions, rewards, states_, dones
