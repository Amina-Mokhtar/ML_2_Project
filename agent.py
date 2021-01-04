from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from tqdm import tqdm
from env import Env
from board import Board

class Agent(object):
    '''
    Handle training the agent
    '''
    def __init__(self, env,board):
        self.__gamma = 0.95                     # learning rate
        self.__eps = 1                          # epsilon
        self.__eps_min = 0.001                  # min epsilon
        self.__eps_decay = 0.995                # 
        self.__batch_size = 64                  # 
        self.__lr = 0.01                        # learning rate
        self.__env = env                        # environment of the agent
        self.__board = board
        self.__memory = deque(maxlen=100000)    # array of past actions
        self.__model = self.__model()
    
    def __model(self):          # initialize model, 1 input, 1 hidden, 1 output later
        model = Sequential()
        model.add(Dense(64, input_shape=(16,), activation='relu')) # self.__env.state_space-6
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.__env.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.__lr))
        return model

    def __remember(self, state, action, reward, next_state, done):      # add previous to memory
        self.__memory.append((state, action, reward, next_state, done))

    def __act(self, state):
        valid_moves = self.__env.valid_moves()
        if np.random.rand() <= self.__eps:
            return np.random.choice(valid_moves)   # instead of action space, use set of valid moves
# how to make it work for below I don't know... Have to restrict the model to predict valid moves... 

        act_values = self.__model.predict(np.reshape(np.array(state), (1,16)))
        
        return np.argmax(act_values[0])

    def __replay(self):                                             # replay and update weights
        if len(self.__memory) < self.__batch_size:                  # if memory is not full yet
            return

        minibatch = random.sample(self.__memory, self.__batch_size) # choose batch_size unique elements from mem 
        states = np.array([i[0] for i in minibatch])                # get states from minibatch
        actions = np.array([i[1] for i in minibatch])               # get actions from minibatch
        rewards = np.array([i[2] for i in minibatch])               # get rewards from minibatch
        next_states = np.array([i[3] for i in minibatch])           # get next_states from minibatch
        dones = np.array([i[4] for i in minibatch])                 # get dones from minibatch

        # states = np.squeeze(states)
        # next_states = np.squeeze(next_states)

        targets = rewards + self.__gamma*(np.amax(self.__model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.__model.predict_on_batch(states)

        ind = np.array([i for i in range(self.__batch_size)])
        targets_full[[ind], [actions]] = targets

        # print(states.shape, targets_full.shape)

        self.__model.fit(states, targets_full, epochs=1, verbose=0)
        if self.__eps > self.__eps_min:
            self.__eps *= self.__eps_decay

    def __print_action(self, action, piece_id):
        if action == 0:
            print(str(piece_id)+" left")
        elif action == 1:
            print(str(piece_id)+" right")
        elif action == 2:
            print(str(piece_id)+" up")
        elif action == 3:
            print(str(piece_id)+" down")

    def __pa2pa(self, piece_action):
        piece_id = int(np.floor(piece_action/ self.__env.npieces))
        action = piece_action % self.__env.npieces
        return piece_id, action

    def train(self, epochs):            # iterate training
        loss = []
        max_moves = 60
         
        for e in tqdm(range(epochs)):                     # iterate number of epochs
            state = self.__env.reset() 
            score = 0
            
            for m in tqdm(range(max_moves)):    # iterate moves until win or max. moves
                piece_action = self.__act(state)
                piece_id, action = self.__pa2pa(piece_action)
                while not self.__env.valid(piece_id,action):
                    piece_action = self.__act(state)
                    piece_id, action = self.__pa2pa(piece_action)
                valid = self.__env.valid(piece_id,action)
                reward, next_state, done = self.__env.step(piece_id, action)
                score += reward - np.log(m+1)             # penalise taking long
                self.__remember(state, piece_action, reward, next_state, done)
                state = next_state
                self.__replay()
                
                text = 'Episode: '+str(e)+'/'+str(epochs) + \
                            ', Move: '+str(m)+', '+ \
                            str(piece_id)+' '+str(action)+' '+str(valid)
                self.__board.draw_board(text)

                if done:
                    print("Dooooooooone:\n")
                    break
            print("Done, or max. moves reached\n")
            loss.append(score)
        return loss