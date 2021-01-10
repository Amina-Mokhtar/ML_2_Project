from numpy import random
from tensorflow.python.framework.ops import disable_tensor_equality
from colors import Colors
import pygame as pg
import numpy as np
import itertools

class Env(object):
    '''
    Handle the environment of the game, using column, row representation
    '''
    def __init__(self, dim, seed=0):
        self.__dim = dim                                        # dimensions of board
        self.__seed = seed                                      # define seed used in generating opponent
        self.__n = dim - 4                                      # dimension of starting square of pieces
        self.__pieces, self.__obst = self.__createPieces()
        self.__maxdist = np.linalg.norm(self.__pieces-np.array([[self.dim-1,0],]*self.npieces),axis=1).sum()

    def __createPieces(self):
        pieces = np.zeros([self.npieces,2])
        obst = np.zeros([self.npieces,2])
        c_range = np.arange(self.__dim - self.__n, self.__dim)
        r_range = np.arange(0, self.__n)

        c = np.repeat(c_range,2) # set range for ...
        r = np.tile(r_range,2)                         # ... player pieces
        for i in range(self.npieces):
            pieces[i] = [r[i],c[i]]

        k = 0
        mem = []
        while k < self.npieces:
            r = random.randint(self.__dim - 1) 
            c = random.randint(self.__dim - 1)
            if (not ((c in c_range and r in r_range) or (c in r_range and r in c_range))):
                if ((r, c) not in mem):
                    mem.append((r, c))
                    obst[k] = [r,c]
                    k += 1
                    
        return pieces, obst
        
    def return_pieces(self):
        return self.__pieces, self.__obst

    def reset(self):
        self.__pieces, self.__obst = self.__createPieces()  # generate new starting position.
        state = self.__get_state()                          # get board state
        return state

    def __done(self):                   # return whether pieces are in finish positions
        x = np.zeros([self.npieces,1])
        y = np.zeros([self.npieces,1])
        for i in range(self.npieces):
            x[i],y[i] = self.__pieces[i]

        Xs = (x >= (self.__dim-self.__n))
        Ys = (y < (self.__n))
        done = Xs.all() and Ys.all()
        partialdone = np.count_nonzero(Xs & Ys)

        target = np.array([[self.dim-1,0],]*self.npieces)
        distance = self.__maxdist - np.linalg.norm(self.__pieces-target,axis=1).sum()

        return done, partialdone, distance

    def valid(self, id, action):                    # determine whether move is valid.
        if (id < 0 or id > (self.npieces - 1)):       # if piece exists
            return False
        x,y = self.__pieces[id]
        rest_pieces = np.vstack((np.delete(self.__pieces,id,0),self.__obst)) # pieces that are not moved
        rest_x = rest_pieces[:,0]
        rest_y = rest_pieces[:,1]
        
        if action == 0:
            for i in range(2*self.npieces-1):
                if ((rest_x[i] == x - 1) and (rest_y[i] == y)):
                    return False
                elif (x - 1 < 0):
                    return False
            return True
        elif action == 1:
            for i in range(2*self.npieces-1):
                if ((rest_x[i] == x + 1) and (rest_y[i] == y)):
                    return False
                elif (x + 1 > self.__dim-1):
                    return False
            return True
        elif action == 2:
            for i in range(2*self.npieces-1):
                if ((rest_y[i] == y - 1) and (rest_x[i] == x)):
                    return False
                elif (y - 1 < 0):
                    return False
            return True
        elif action == 3:
            for i in range(2*self.npieces-1):
                if ((rest_y[i] == y + 1) and (rest_x[i] == x)):
                    return False
                elif (y + 1 > self.__dim-1):
                    return False
            return True

    def valid_moves(self): # return matrix of valid moves
        '''
        Compute valid moves
        '''
        valid_moves = np.zeros((self.dim,self.dim,self.npieces))
        for i in range(self.npieces):              # find valid moves for single move
            for a in range(4):
                if self.valid(i,a):
                    if a == 0:
                        x = self.__pieces[i,0] - 1  # move piece if valid left
                        y = self.__pieces[i,1]
                    if a == 1:
                        x = self.__pieces[i,0] + 1  # move piece right
                        y = self.__pieces[i,1]
                    if a == 2:
                        x = self.__pieces[i,0]
                        y = self.__pieces[i,1] - 1  # move piece up
                    if a == 3:
                        x = self.__pieces[i,0]
                        y = self.__pieces[i,1] + 1  # move piece down
                    valid_moves[int(y),int(x),i] = 1
        return valid_moves

    # def __move(self, id, action):
    #     if not self.valid(id, action):       # if piece exists
    #         return 
    #     elif action == 0:
    #         self.__pieces[id,0] -= 1  # move piece if valid left
    #     elif action == 1:
    #         self.__pieces[id,0] += 1  # move piece right
    #     elif action == 2:
    #         self.__pieces[id,1] -= 1  # move piece up
    #     elif action == 3:
    #         self.__pieces[id,1] += 1  # move piece down

    def __get_state(self):      # board state with player and obstacle pos.
        state = np.zeros((self.__dim,self.__dim,2))
        for i in range(self.npieces):
            x = self.__pieces.astype(int)[i,0]
            y = self.__pieces.astype(int)[i,1]
            state[x,y,0] = 1
            x = self.__obst.astype(int)[i,0]
            y = self.__obst.astype(int)[i,1]
            state[y,x,1] = 1
        return state.flatten()

    def step(self, action):           # one training step
        y, x, piece_id = np.unravel_index(action,(self.dim,self.dim,self.npieces))
        self.__pieces[piece_id] = (x,y)
        done, partial, distance = self.__done()           # return new x and y of pieces
        reward = 0
        
        # reward += distance/10
        # reward += partial/4
        if done:
            reward += 1
        
        
        # if (action == 0 or action == 3):    # give reward for movement
        #     reward -= 10                     # negative for moving left and dwon
        # else:
        #     reward += 2                     # positive for moving right and up
        
        state = self.__get_state()          # get new state
        return reward, state, done

    @property
    def npieces(self):
        return self.__n**2

    @property
    def state_space(self):
        return (self.__dim**2)*2       # Number of possible game states, x and y pos. of player and opponent

    @property
    def action_space(self):
        return (self.__dim**2)*self.npieces  # Action space, 4 moves per piece

    @property
    def dim(self):
        return self.__dim
