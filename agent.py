from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, Activation, add, MaxPooling2D
from tqdm import tqdm
from collections import deque
import numpy as np
import math
import random

class Agent(object):
    def __init__(self, env):
        self.__gamma = 0.75
        self.__eps = 1.0
        self.__eps_min = 0.01
        self.__eps_decay = 0.99995
        self.__batch_size = 128
        self.__lr = 0.0001
        self.__env = env
        self.__memory = deque(maxlen=10000000)
        self.__model = self.__model()

    def __model(self):
        model = Sequential()
        model.add(Conv2D(16, (1, 3), activation='relu', input_shape=self.__env.state_space))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(16, (3, 1), activation='relu'))
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(16, activation='linear'))
        model.add(Dense(self.__env.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.__lr))
        # print(model.summary())
        return model

    def __remember(self, state, action, reward, next_state, id, done):
        self.__memory.append((state, action, reward, next_state, id, done))

    def __act(self, state):
        ID_mask = self.__env.getIDMask()

        if np.random.rand() <= self.__eps:
            id = np.random.randint(self.__env.npieces)
            moves, jumps = self.__env.availableMoves(id)
            while len(moves) == 0 and len(jumps) == 0:
                id = np.random.randint(self.__env.npieces)
                moves, jumps = self.__env.availableMoves(id)
            action = np.random.choice(moves + jumps, 1)
            # print("rand")
            # print(id)
            return action[0], np.expand_dims(ID_mask[id], axis=0)

        best_action = None
        best_value = -math.inf
        best_id = None

        for id in range(self.__env.npieces):
            model_state = np.append(state, np.expand_dims(ID_mask[id], axis=0), axis=0)

            act_values = self.__model.predict_on_batch(np.expand_dims(model_state, axis=0))
            temp_max = np.max(act_values[0])
            if temp_max > best_value:
                best_value = temp_max
                best_id = np.expand_dims(ID_mask[id], axis=0)
                best_action = np.argmax(act_values[0])

        # print("best")
        # print(best_id)
        return best_action, best_id

    def __replay(self):
        if len(self.__memory) < self.__batch_size:
            return

        learning_rate = 0.5
        minibatch = random.sample(self.__memory, self.__batch_size)
        states = np.array([np.append(i[0], i[4], axis=0) for i in minibatch])
        next_states = np.array([np.append(i[3], i[4], axis=0) for i in minibatch])

        current_qs_list = self.__model.predict_on_batch(states)
        future_qs_list = self.__model.predict_on_batch(next_states)

        X = []
        Y = []
        for index, (state, action, reward, _, id, done) in enumerate(minibatch):
            if not done:
                max_future_q = reward + self.__gamma * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(np.append(state, id, axis=0))
            Y.append(current_qs)

        self.__model.fit(np.array(X), np.array(Y), batch_size=self.__batch_size, verbose=0, shuffle=True)

        if self.__eps > self.__eps_min:
            self.__eps *= self.__eps_decay

    def train(self, epochs):
        loss = []
        max_moves = 2000
        dones = 0
        move = []
        for e in tqdm(range(epochs)):
            state = self.__env.reset()
            score = 0
            for i in range(max_moves):
                action, id = self.__act(state)
                reward, next_state, done = self.__env.step(action, self.__env.mask2num(id) - 1)
                score += reward
                self.__env.updateEnv(e, reward, self.__eps, score, dones)
                self.__remember(state, action, reward, next_state, id, done)
                state = next_state
                self.__replay()
                if done:
                    dones += 1
                    move.append((i, e))
                    break
            loss.append(score)
        print(dones)
        print(loss)
        return loss, move