from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import math
import random

class Agent(object):
    def __init__(self, env):
        self.__gamma = 0.9
        self.__eps = 0.1
        self.__eps_min = 0.01
        self.__eps_decay = 0.995
        self.__batch_size = 64
        self.__lr = 0.001
        self.__env = env
        self.__memory = deque(maxlen=100000)
        self.__model = self.__model()

    def __model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(6*6+1,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.__env.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.__lr))
        return model

    def __remember(self, state, action, reward, next_state, id, done):
        self.__memory.append((state, action, reward, next_state, id, done))

    def __act(self, state):
        # temp_id = 1
        # if np.random.rand() <= self.__eps:
        #     moves, jumps = self.__env.availableMoves(temp_id)
        #     action = np.random.choice(moves + jumps, 1)
        #     return action[0]

        best_action = None
        best_value = -math.inf
        best_id = None

        for id in range(4): # change to a variable, dont leave loose ends...
            model_state = np.append(state, id)
            act_values = self.__model.predict(np.reshape(np.array(model_state), (1, 6*6+1)))
            temp_max = np.max(act_values[0])
            if temp_max > best_value:
                best_value = temp_max
                best_id = id
                best_action = np.argmax(act_values[0])

        return best_action, best_id

    def __replay(self):
        if len(self.__memory) < self.__batch_size:
            return

        minibatch = random.sample(self.__memory, self.__batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        ids = np.array([i[4] for i in minibatch])
        dones = np.array([i[5] for i in minibatch])

        next_states_temp = []
        states_temp = []
        for i in range(len(next_states)):
            states_temp.append(np.append(states[i], ids[i]))
            next_states_temp.append(np.append(next_states[i], ids[i]))


        next_states = np.array(next_states_temp)
        states = np.array(states_temp)
        # print("shaaaaaaape")
        # print(next_states.shape)
        # print(next_states.shape)

        targets = rewards + self.__gamma*(np.amax(self.__model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.__model.predict_on_batch(states)

        ind = np.array([i for i in range(self.__batch_size)])
        targets_full[[ind], [actions]] = targets

        self.__model.fit(states, targets_full, epochs=1, verbose=0)
        if self.__eps > self.__eps_min:
            self.__eps *= self.__eps_decay

    def train(self, epochs):
        loss = []
        max_moves = 100
        re = []

        for e in range(epochs):
            state = self.__env.reset()
            score = 0
            print(e)
            for m in range(max_moves):
                action, id = self.__act(state)
                reward, next_state, done = self.__env.step(action, id)
                re.append(reward)

                score += reward
                self.__remember(state, action, reward, next_state, id, done)
                state = next_state
                self.__replay()
                if done:
                    print("Dooooooooone:")
                    break
            loss.append(score)
        print(loss)
        #print(re)
        return loss