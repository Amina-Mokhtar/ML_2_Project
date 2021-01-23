from keras.optimizers import Adam
from buffer import ReplayBuffer
from nn import DuelingDeepQNet
import tensorflow as tf
import numpy as np
import math

class AgentDuelingDeepQNet(object):
    def __init__(self, env, lr, gamma, batch_size, eps_dec=0.9996, mem_size=1000000, replace=100):
        self.env = env
        self.action_space = [i for i in range(self.env.action_space)]
        self.gamma = gamma
        self.eps = 1
        self.eps_dec = eps_dec
        self.eps_min = 0.01
        self.batch_size = batch_size
        self.replace = replace
        self.memory = ReplayBuffer(mem_size, self.env.state_space)
        self.q_eval = DuelingDeepQNet(self.env.state_space, self.env.action_space)
        self.q_next = DuelingDeepQNet(self.env.state_space, self.env.action_space)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def remember(self, state, action, reward, new_state, piece_id, done):
        self.memory.push(state, action, reward, new_state, piece_id, done)

    def decay_eps(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec

    def update_network_parameters(self):
        if self.memory.memory_cntr % self.replace == 0:
            self.q_next = self.q_eval
            # self.q_target.model.set_weights(self.q_eval.model.get_weights())

    def act(self, state):
        id_mask = self.env.getIDMask()

        if np.random.rand() <= self.eps:
            piece_id = np.random.randint(self.env.npieces)
            moves, jumps = self.env.availableMoves(piece_id)
            while len(moves) == 0 and len(jumps) == 0:
                piece_id = np.random.randint(self.env.npieces)
                moves, jumps = self.env.availableMoves(piece_id)
            action = np.random.choice(moves + jumps, 1)
            return action[0], np.expand_dims(id_mask[piece_id], axis=0)

        best_action = None
        best_value = -math.inf
        best_id = None

        for piece_id in range(self.env.npieces):
            model_state = np.append(state, np.expand_dims(id_mask[piece_id], axis=0), axis=0)
            act_values = self.q_eval.advantage(np.expand_dims(model_state, axis=0))
            temp_max = np.max(act_values[0])
            if temp_max > best_value:
                best_value = temp_max
                best_id = np.expand_dims(id_mask[piece_id], axis=0)
                best_action = np.argmax(act_values[0])

        return best_action, best_id

    def replay(self):
        if self.memory.memory_cntr > self.batch_size:

            states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

            q_pred = self.q_eval.call(states)
            q_next = tf.math.reduce_max(self.q_next.call(new_states), axis=1, keepdims=True).numpy()
            q_target = np.copy(q_pred)

            for idx, done in enumerate(dones):
                q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx] * (1 - done)

            # self.q_eval.train_on_batch(states, q_target)
            self.q_eval.fit(states, q_target, verbose=0)
            self.decay_eps()
            self.update_network_parameters()

    def play(self, epochs, max_moves):
        loss, moves = [], []
        dones = 0
        for e in range(epochs):
            state = self.env.reset()
            score = 0
            for i in range(max_moves):
                action, piece_id = self.act(state)
                reward, next_state, done = self.env.stepEnv(action, self.env.mask2num(piece_id) - 1)
                score += reward
                self.env.updateEnv(e, reward, self.eps, score, dones)
                self.remember(state, action, reward, next_state, piece_id, done)
                state = next_state
                self.replay()
                if done:
                    dones += 1
                    moves.append((i, e))
                    break
            loss.append(score)
        return loss, moves