from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random

class Agent(object):
    def __init__(self, env):
        self.__gamma = 0.95
        self.__eps = 1
        self.__eps_min = 0.01
        self.__eps_decay = 0.995
        self.__batch_size = 64
        self.__lr = 0.001
        self.__env = env 
        self.__memory = deque(maxlen=100000)
        self.__model = self.__model() 

    def __model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(10,), activation='relu')) # self.__env.state_space-6
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.__env.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.__lr))
        return model

    def __remember(self, state, action, reward, next_state, done):
        self.__memory.append((state, action, reward, next_state, done))

    def __act(self, state):
        if np.random.rand() <= self.__eps:
            return random.randrange(self.__env.action_space)

        act_values = self.__model.predict(np.reshape(np.array(state), (1,10)))
        return np.argmax(act_values[0])

    def __replay(self):
        if len(self.__memory) < self.__batch_size:
            return

        minibatch = random.sample(self.__memory, self.__batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # states = np.squeeze(states)
        # next_states = np.squeeze(next_states)

        targets = rewards + self.__gamma*(np.amax(self.__model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.__model.predict_on_batch(states)

        ind = np.array([i for i in range(self.__batch_size)])
        targets_full[[ind], [actions]] = targets

        print(states.shape, targets_full.shape)

        self.__model.fit(states, targets_full, epochs=1, verbose=0)
        if self.__eps > self.__eps_min:
            self.__eps *= self.__eps_decay

    def foo(self, action):
        if action == 0:
            print("left")
        elif action == 1:
            print("right")
        elif action == 2:
            print("up")
        elif action == 3:
            print("down")

    def train(self, epochs):
        loss = []
        max_moves = 50
         
        for e in range(epochs):
            state = self.__env.reset() 
            score = 0
            for m in range(max_moves):
                action = self.__act(state)
                self.foo(action)
                reward, next_state, done = self.__env.step(action)
                score += reward
                self.__remember(state, action, reward, next_state, done)
                state = next_state
                self.__replay()
                if done:
                    print("Dooooooooone:")
                    break
            loss.append(score)
        return loss