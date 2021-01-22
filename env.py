from matplotlib.pyplot import pie
import numpy as np
from numpy import random
from debugs import Debug

class Env(object):
    '''
    Handle the environment of the game, using column, row representation
    '''
    def __init__(self, dim, seed=0):
        self.__dim = dim                                        # dimensions of board
        self.__seed = seed                                      # define seed used in generating opponent
        self.__n = 2                                      # dimension of starting square of pieces
        self.__pieces, self.__obst = self.__createPieces()
        self.__maxdist = np.linalg.norm(self.__pieces-np.array([[self.dim-1,0],]*self.npieces),axis=1).sum()
        self.__valid_moves = np.zeros((self.dim,self.dim,self.npieces))
        self.__get_valid_moves()

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

        target = np.array([[self.dim-2,1],]*self.npieces) # [dim-2,1]
        # distance = np.linalg.norm(self.__pieces-target,axis=1).sum() # norm
        distance = abs(self.__pieces-target).sum() # coty-block distance

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

    def __get_valid_moves(self): # return matrix of valid moves
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
        
        pieces = np.vstack((self.__pieces,self.__obst)).tolist()
        for i in range(self.npieces):       # find valid jumps
            x,y = self.__pieces[i]
            targets = [[x,y]]
            begin_targets = [[x,y]]
            done = False
            while done == False:
                for (x,y) in begin_targets:
                    targets_new = []
                    for (xstep,ystep) in np.array([[0,1],[1,0],[-1,0],[0,-1]]):
                        xtarget, ytarget = x+2*xstep , y+2*ystep
                        x_over, y_over = x+xstep,y+ystep
                        if ( ([x_over, y_over] in pieces) and                 # piece to jump over
                                            ([xtarget,ytarget] not in pieces) and          # no piece at target
                                            (0 <= xtarget < self.dim) and (0 <= ytarget < self.dim) and # target on board
                                            ([xtarget,ytarget] not in targets) ):             # new target 
                            targets_new.append([xtarget,ytarget])
                            targets.append([xtarget,ytarget])
                if len(targets_new) == 0:
                    done = True
                    break
                else:
                    begin_targets = targets_new
            for x,y in targets:
                if [x,y] not in pieces:
                    valid_moves[int(y),int(x),i] = 1
        self.__valid_moves = valid_moves
        # Debug().save_all(self.__pieces,self.__obst,valid_moves=self.valid_moves)
        return 

    def __get_state(self):      # board state with player and obstacle pos.
        state = np.zeros((self.__dim,self.__dim,2))
        for i in range(self.npieces):
            x = self.__pieces.astype(int)[i,0]
            y = self.__pieces.astype(int)[i,1]
            state[x,y,0] = i
            x = self.__obst.astype(int)[i,0]
            y = self.__obst.astype(int)[i,1]
            state[y,x,1] = 1
        return state

    def __move_dist(self, pieces_prev,id):
        cityblock = (self.__pieces[id] - pieces_prev[id])*np.array([1,-1])
        norm = np.linalg.norm(cityblock)
        move_dist = norm*np.sign(cityblock.sum())
        return move_dist

    def step(self, action):           # one training step
        pieces_prev = np.copy(self.__pieces)
        reward = 0
        y, x, piece_id = np.unravel_index(action,(self.dim,self.dim,self.npieces)) # pylint: disable=unbalanced-tuple-unpacking
        if self.valid_moves[y,x,piece_id] == 0: # Move invalid
            reward += -1
            valid = False
        else:
            valid = True
            self.__pieces[piece_id] = (x,y)         # move piece
            self.__get_valid_moves()                # recompute valid moves
        state = self.__get_state()          # get new state

        done, partial, distance = self.__done() # return new x and y of pieces
        # reward += partial/4
        if done:
            reward += 10
        else:
            # reward += -1
            # reward -= distance/10
            move_dist = self.__move_dist(pieces_prev,piece_id)
            if move_dist>0:
                reward += move_dist
            else:
                reward += move_dist*0.01

        return reward, state, done, valid

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

    @property
    def valid_moves(self):
        return self.__valid_moves
