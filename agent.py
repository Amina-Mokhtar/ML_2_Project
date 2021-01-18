from keras import Sequential, regularizers
from keras.models import Model as KerasModel
from keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, Activation, add, MaxPooling2D
from keras.optimizers import Adam, SGD
from collections import deque
import numpy as np
import random as rnd
import tensorflow as tf
# from tensorflow.python.keras.utils.generic_utils import validate_config
from tqdm import tqdm
from env import *
from board import *
from debugs import *

class Agent(object):
    '''
    Handle training the agent
    Model Options:
     - default
     - conv
     - convCC
    '''
    def __init__(self, env,board,model_type='default',non_valid=True):
        self.__gamma = 0.75                     # discount rate for future reward
        self.__eps = 1                          # epsilon: beginning exploration prob.
        self.__eps_min = 0.01                  # min epsilon
        self.__eps_decay = 0.99995                # epsilon decay
        self.__batch_size = 128                  # batch size for replay
        self.__lr = 0.001                       # learning rate: proportion of new value
        self.__decay = 6e-3                     # weight decay constant
        self.__env = env                        # environment of the agent
        self.__board = board
        self.__memory = deque(maxlen=10000000)    # array of past actions
        self.__modeltype = model_type
        self.__model = self.__model_init()
        self.__non_valid = non_valid
    
    def __model_init(self):          # initialize model, 1 input, 1 hidden, 1 output later
        if self.__modeltype == 'default':
            model = Sequential()
            model.add(Dense(64, input_shape=(self.__env.state_space,), activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.__env.action_space, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.__lr))
        if self.__modeltype == 'convCC':
            model = self.__build_convCC()
        if self.__modeltype == 'conv':
            model = Sequential()
            model.add(Conv2D(16, (1, 3), activation='relu', input_shape=(self.__env.dim,self.__env.dim,2)))
            # model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(16, (3, 1), activation='relu'))
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(self.__env.action_space, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.__lr))
        # print(model.summary())
        return model

    def __remember(self, state, action, reward, next_state, done):      # add previous to memory
        self.__memory.append((state, action, reward, next_state, done))

    def __act(self, state):
        valid_moves = self.__env.valid_moves.flatten()
        if np.random.rand() <= self.__eps:          # choose any valid move
            action = np.random.choice(np.arange(self.__env.action_space)[valid_moves == 1])
        else:
            if self.__modeltype == 'default':
                state = state.reshape(1,self.__env.state_space)
            else:
                state = state.reshape((1,)+state.shape)
        
            if self.__non_valid == True:
                ############## Choose any move ###########################################
                action = np.argmax(self.__model.predict(state))
                ########### OR Choose only valid moves ###################################
            else:
                actions = self.__model.predict(state)
                actions_valid = actions.reshape((self.__env.action_space,)) * valid_moves
                action = np.random.choice(np.arange(self.__env.action_space)[actions_valid==actions_valid.max()]) # choose random action of all valid
                ##########################################################################

        return action

    def __replay(self):                                             # replay and update weights
        if len(self.__memory) < self.__batch_size:                  # if memory is not full yet
            return

        minibatch = rnd.sample(self.__memory, self.__batch_size) # choose batch_size unique elements from mem 
        states = np.array([i[0] for i in minibatch])                # get states from minibatch
        actions = np.array([i[1] for i in minibatch])               # get actions from minibatch
        rewards = np.array([i[2] for i in minibatch])               # get rewards from minibatch
        next_states = np.array([i[3] for i in minibatch])           # get next_states from minibatch
        dones = np.array([i[4] for i in minibatch])                 # get dones from minibatch

        if self.__modeltype == 'default':
            states = states.reshape(self.__batch_size,self.__env.state_space)
            next_states = states.reshape(self.__batch_size,self.__env.state_space)
        # states = np.squeeze(states)
        # next_states = np.squeeze(next_states)

        if self.__non_valid == True:
            ############## Choose any move ###########################################
            action = np.argmax(self.__model.predict_on_batch(next_states), axis=1)
        else:
            ########### OR Choose only valid moves ###################################
            actions_new = self.__model.predict_on_batch(next_states)
            valids = self.__env.valid_moves.flatten()
            valid_moves = np.tile(valids.T,(self.__batch_size,1))
            actions_valid = actions_new * valid_moves
            action = np.zeros((self.__batch_size,))
            for i in range(self.__batch_size):
                action[i] = np.random.choice(np.arange(self.__env.action_space)[actions_valid[i]==actions_valid[i].max()])
            ##########################################################################


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
        win = []
        eps = []
        max_moves = moves
         
        for e in tqdm(range(epochs)):         # iterate number of epochs
            state = self.__env.reset() 
            score = 0
            
            for m in tqdm(range(max_moves)):    # iterate moves until win or max. moves
                action = self.__act(state)
                reward, next_state, done, valid = self.__env.step(action)
                score += reward
                self.__remember(state, action, reward, next_state, done)
                state = next_state
                self.__replay()

                # y, x, piece_id = np.unravel_index(action,(self.__env.dim,self.__env.dim,self.__env.npieces))
                # valid = self.__env.valid_moves[y,x,piece_id]
                text = str(valid) + \
                        ", Episode: {:<4d}".format(e) +'/'+str(epochs) + \
                        ", Move: {:<4d}".format(m) + \
                        ', Reward: '+ "{:.2f}".format(reward)
                        # str(piece_id)+' '+str(x)+','+str(y) + \
                text2 = "eps: {:.2f}".format(self.__eps)
                self.__board.draw_board(text,text2)

                if done:
                    print("\nDooooooooone:\n")
                    break
            # print("\nDone, or max. moves reached\n")
            loss.append(score)
            win.append(int(done))
            eps.append(self.__eps)
            
            # Debug().save_all(self.__env.__pieces,self.__env.__obst,valid_moves=self.valid_moves,fname='endEp_'+str(e),action=action)

        return loss, win, eps

    def save_model(self,fname="model"):
        self.__model.save_weights("models/"+fname+".h5")
        return




############## From Chinese Checkers Paper ############

    def __build_convCC(self):
        main_input = Input(shape=(self.__env.dim,self.__env.dim,2))
        regularizer = regularizers.l2(self.__decay)

        x = Conv2D(filters=64, kernel_size=3, kernel_regularizer=regularizer, padding='valid')(main_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)
        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)
        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)

        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)
        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)
        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)

        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)
        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)
        x = self.residual_block(x, [32, 32, 64], kernel_size=3, regularizer=regularizer)

        policy = self.policy_head(x, regularizer)

        model = KerasModel(inputs=[main_input], outputs=[policy])
        # model.compile(loss=self.softmax_cross_entropy_with_logits, optimizer=SGD(lr=self.__lr, momentum=0.9, nesterov=True))
        model.compile(loss='mse', optimizer=SGD(lr=self.__lr, momentum=0.9, nesterov=True))

        return model

    def softmax_cross_entropy_with_logits(y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

    def policy_head(self, head_input, regularizer):
        x = Conv2D(filters=16, kernel_size=1, kernel_regularizer=regularizer)(head_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.__env.action_space,
                use_bias=True,
                activation='linear',
                kernel_regularizer=regularizer,
                name='policy_head')(x)
        return x


    def residual_block(self, block_input, filters, kernel_size, regularizer):
        '''
        Residual block setup code referenced from Keras
        https://github.com/keras-team/keras
        '''
        x = Conv2D(filters=filters[0]
                 , kernel_size=1
                 , kernel_regularizer=regularizer)(block_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters[1]
                 , kernel_size=kernel_size
                 , padding='same'
                 , kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters[2]
                 , kernel_size=1
                 , kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)

        x = add([x, block_input])
        x = Activation('relu')(x)
        return x


    def conv_block(self, block_input, filters, kernel_size, regularizer):
        '''
        Conv block setup code referenced from Keras
        https://github.com/keras-team/keras
        '''
        x = Conv2D(filters=filters[0]
                 , kernel_size=1
                 , kernel_regularizer=regularizer)(block_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters[1]
                 , kernel_size=kernel_size
                 , padding='same'
                 , kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters[2]
                 , kernel_size=1
                 , kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)

        shortcut = Conv2D(filters=filters[2]
                        , kernel_size=1
                        , kernel_regularizer=regularizer)(block_input)
        shortcut = BatchNormalization()(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x 