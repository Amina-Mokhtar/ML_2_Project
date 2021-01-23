import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.memory_size = max_size
        self.memory_cntr = 0
        self.priority = deque(maxlen=max_size)

        self.state_memory = np.zeros((self.memory_size, *input_shape),
            dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, *input_shape),
            dtype=np.float32)

        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def push(self, state, action, reward, next_state, piece_id, done):
        index = self.memory_cntr % self.memory_size
        self.state_memory[index] = np.append(state, piece_id, axis=0)
        self.next_state_memory[index] = np.append(next_state, piece_id, axis=0)
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.memory_cntr, self.memory_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal


    def prioritize(self, state, action, reward, next_state, id, done, model, gamma, alpha=0.6):
        model_next_state = np.append(next_state, id, axis=0)
        model_state = np.append(state, id, axis=0)
        result_next_state = np.squeeze(model.call(np.expand_dims(model_next_state, axis=0)))
        result_state = np.squeeze(model.call(np.expand_dims(model_state, axis=0)))

        q_next = reward + gamma * np.max(result_next_state)
        q = result_state[action]
        p = (np.abs(q_next - q) + (np.e ** -10)) ** alpha

        self.priority.append(p)
        self.push(state, action, reward, next_state, id, done)

    def get_priority_sample(self, batch_size):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum
        batch = random.choices(range(len(prob)), k=batch_size, weights=prob)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, next_states, done