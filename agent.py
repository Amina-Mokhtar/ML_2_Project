from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from numpy.core.fromnumeric import amax
from tensorflow.python.keras.utils.generic_utils import validate_config
from tqdm import tqdm
from env import Env
from board import Board
from debugs import Debug

class Agent(object):
    '''
    Handle training the agent
    '''
    def __init__(self, env,board):
        self.__gamma = 0.95                     # discount rate for future reward
        self.__eps = 1                          # epsilon: beginning exploration prob.
        self.__eps_min = 0.001                  # min epsilon
        self.__eps_decay = 0.999                # epsilon decay
        self.__batch_size = 64                  # batch size for replay
        self.__lr = 0.001                       # learning rate: proportion of new value
        self.__env = env                        # environment of the agent
        self.__board = board
        self.__memory = deque(maxlen=100000)    # array of past actions
        self.__model = self.__model_init()
    
    def __model_init(self):          # initialize model, 1 input, 1 hidden, 1 output later
        model = Sequential()
        model.add(Dense(64, input_shape=(self.__env.state_space,), activation='relu')) # self.__env.state_space-6
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.__env.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.__lr))
        return model

    def __remember(self, state, action, reward, next_state, done):      # add previous to memory
        self.__memory.append((state, action, reward, next_state, done))

    def __act(self, state):
        valid_moves = self.__env.valid_moves().flatten()
        if np.random.rand() <= self.__eps:          # choose any valid move
            action = np.random.choice(np.arange(self.__env.action_space)[valid_moves == 1])
        else:
            actions = self.__model.predict(np.reshape(state, (1,self.__env.state_space)))
            actions_valid = actions.reshape((self.__env.action_space,)) * valid_moves
            action = np.random.choice(np.arange(self.__env.action_space)[actions_valid==actions_valid.max()]) # choose random action of all valid

            # action = np.argmax(self.__model.predict(np.reshape(state, (1,self.__env.state_space))))

        return action

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

        action_ref = np.argmax(self.__model.predict_on_batch(next_states), axis=1)

        valids = self.__env.valid_moves().flatten()
        valid_moves = np.tile(valids.T,(self.__batch_size,1))
        actions_new = self.__model.predict_on_batch(next_states)
        actions_valid = actions_new * valid_moves
        action = np.zeros((self.__batch_size,))
        for i in range(self.__batch_size):
            action[i] = np.random.choice(np.arange(self.__env.action_space)[actions_valid[i]==actions_valid[i].max()])


        targets = rewards + self.__gamma*action*(1-dones)
        targets_full = self.__model.predict_on_batch(states)

        ind = np.array([i for i in range(self.__batch_size)])
        targets_full[[ind], [actions]] = targets

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

    def train(self, epochs,moves):            # iterate training
        loss = []
        max_moves = moves
         
        for e in tqdm(range(epochs)):                     # iterate number of epochs
            state = self.__env.reset() 
            score = 0
            
            for m in tqdm(range(max_moves)):    # iterate moves until win or max. moves
                action = self.__act(state)
                reward, next_state, done = self.__env.step(action)
                score += reward
                self.__remember(state, action, reward, next_state, done)
                state = next_state
                self.__replay()
                
                y, x, piece_id = np.unravel_index(action,(self.__env.dim,self.__env.dim,self.__env.npieces))
                text = 'Episode: '+str(e)+'/'+str(epochs) + \
                            ', Move: '+str(m)+', '+ \
                            ', Reward: '+str(reward)+', '+\
                            str(piece_id)+' '+str(x)+','+str(y)
                self.__board.draw_board(text)

                if done:
                    print("\nDooooooooone:\n")
                    break
            print("\nDone, or max. moves reached\n")
            loss.append(score)
        return loss