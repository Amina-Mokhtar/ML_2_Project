from numpy import random
from colors import Colors
import pygame as pg
import numpy as np
import itertools

class Env(object):
    def __init__(self, width, height, dim, seed=0):
        self.__screen = pg.display.set_mode((width, height))    # pygame screen object
        self.__font = pg.font.SysFont(None, 24)
        self.__length = 30                                      # size of pieces
        self.__step = 75                                        # width of board squares (px)
        self.__dim = dim                                        # dimensions of board
        self.__seed = seed                                      # define seed used in generating opponent
        self.__n = dim - 4                                      # dimension of starting square of pieces
        self.__color = itertools.cycle((Colors.WHITE, Colors.BLACK))    # board colours
        self.__board_length = self.__dim * self.__step          # board size in px
        self.__X = (width - self.__board_length) / 2            # x coordinate of top-left corner of board
        self.__Y = (height - self.__board_length) / 2           # y coordinate of top-left corner of board
        self.__pieces, self.__obst = self.__createPieces()      # The player has pieces, and the opponent is the obstacles.
        self.__background = self.__drawBoard()                  # board object, draws the board

    def __drawBoard(self):
        background = pg.Surface((self.__board_length, self.__board_length)) # pygame surface of board
        for y in range(0, self.__board_length, self.__step):
            for x in range(0, self.__board_length, self.__step):
                rect = (x, y, self.__step, self.__step)                     
                pg.draw.rect(background, next(self.__color), rect)          # draw rectangles to create board pattern
            next(self.__color)
        return background

    def __pos2coord(self, row, col):    # convert row/column to coordinates in window
        x = (self.__step / 2) + row * self.__step + self.__X - self.__length / 2
        y = (self.__step / 2) + col * self.__step + self.__Y - self.__length / 2
        return x, y

    def __createPieces(self):
        rects1 = []
        rects2 = []
        if self.__dim - self.__n == 4:
            c_range = range(self.__dim - self.__n, self.__dim)  # set range for ...
            r_range = range(0, self.__n)                        # ... player pieces

            for c in c_range:
                for r in r_range:
                    x, y = self.__pos2coord(r, c)
                    rect = pg.rect.Rect(x, y, self.__length, self.__length)
                    rects1.append(rect)     # Create list of objects for pieces

            k = 0
            mem = []
            while k < 2*self.__n:                   # generate random unique locations of opponent
                r = random.randint(self.__dim - 1)  # TODO: add possibility for a seed to be able to replicate random boards
                c = random.randint(self.__dim - 1)
                if (not ((c in c_range and r in r_range) or (c in r_range and r in c_range))):
                    if ((r, c) not in mem):
                        mem.append((r, c))
                        x, y = self.__pos2coord(r, c)
                        rect = pg.rect.Rect(x, y, self.__length, self.__length)
                        rects2.append(rect)
                        k += 1
                               
        return rects1, rects2  

    def reset(self):
        self.__pieces, self.__obst = self.__createPieces()  # generate new starting position.
        state = self.__get_state()                          # get board state
        return state

    def __done(self):                   # return whether pieces are in finish positions
        # temp_id = 1                     # currently only checks for piece 1
        # x = self.__pieces[temp_id].x
        # y = self.__pieces[temp_id].y
        # return (x >= self.__step*(self.__dim-self.__n) + self.__X and y <= self.__step*self.__n + self.__Y)

        x = np.zeros([self.npieces,1])
        y = np.zeros([self.npieces,1])
        for i in range(self.npieces):
            x[i] = self.__pieces[i].x
            y[i] = self.__pieces[i].y

        Xs = (x >= (self.__step*(self.__dim-self.__n) + self.__X))
        Ys = (y <= (self.__step*self.__n + self.__Y))
        done = Xs.all() and Ys.all()
        partialdone = np.count_nonzero(Xs & Ys)
        return done, partialdone


    def valid(self, id, action):                    # determine whether move is valid.
        if (id < 0 or id > (2*self.__n - 1)):       # if piece exists
            return False
        
        rest_pieces = [item for i, item in enumerate(self.__pieces) if i not in [id]] + self.__obst # pieces that are not moved

        if action == 0:
            for item in rest_pieces:
                if (item.x == self.__pieces[id].x - self.__step and item.y == self.__pieces[id].y or    # there is a piece in the new position
                    self.__pieces[id].x - self.__step < self.__X):                                      # new position is out of bounds
                    return False
            return True
        elif action == 1:
            for item in rest_pieces:
                if (item.x == self.__pieces[id].x + self.__step and item.y == self.__pieces[id].y or 
                    self.__pieces[id].x + self.__step > self.__board_length + self.__X):
                    return False
            return True
        elif action == 2:
            for item in rest_pieces:
                if (item.y == self.__pieces[id].y - self.__step and item.x == self.__pieces[id].x or
                    self.__pieces[id].y - self.__step < self.__Y):
                    return False
            return True
        elif action == 3:
            for item in rest_pieces:
                if (item.y == self.__pieces[id].y + self.__step and item.x == self.__pieces[id].x or
                    self.__pieces[id].y + self.__step > self.__board_length + self.__Y):
                    return False
            return True

    def move(self, id, action):
        if not self.valid(id, action):       # if piece exists
            return 
        elif action == 0:
            self.__pieces[id].x -= self.__step  # move piece if valid left
        elif action == 1:
            self.__pieces[id].x += self.__step  # move piece right
        elif action == 2:
            self.__pieces[id].y -= self.__step  # move piece down
        elif action == 3:
            self.__pieces[id].y += self.__step  # move piece up

    # def move(self, id, action):
    #     if (id < 0 or id > (2*self.__n - 1)):       # if piece exists
    #         return 
        
    #     rest_pieces = [item for i, item in enumerate(self.__pieces) if i not in [id]] + self.__obst # pieces that are not moved

    #     if action == 0:
    #         for item in rest_pieces:
    #             if (item.x == self.__pieces[id].x - self.__step and item.y == self.__pieces[id].y or    # there is a piece in the new position
    #                 self.__pieces[id].x - self.__step < self.__X):                                      # new position is out of bounds
    #                 return 
    #         self.__pieces[id].x -= self.__step  # move piece if valid
    #     elif action == 1:
    #         for item in rest_pieces:
    #             if (item.x == self.__pieces[id].x + self.__step and item.y == self.__pieces[id].y or 
    #                 self.__pieces[id].x + self.__step > self.__board_length + self.__X):
    #                 return 
    #         self.__pieces[id].x += self.__step  # move piece
    #     elif action == 2:
    #         for item in rest_pieces:
    #             if (item.y == self.__pieces[id].y - self.__step and item.x == self.__pieces[id].x or
    #                 self.__pieces[id].y - self.__step < self.__Y):
    #                 return 
    #         self.__pieces[id].y -= self.__step  # move piece
    #     elif action == 3:
    #         for item in rest_pieces:
    #             if (item.y == self.__pieces[id].y + self.__step and item.x == self.__pieces[id].x or
    #                 self.__pieces[id].y + self.__step > self.__board_length + self.__Y):
    #                 return 
    #         self.__pieces[id].y += self.__step  # move piece

    def __get_state(self):      # board state with player and obstacle pos.
        temp_id = 1             # piece 1
        pieces_pos = [self.__pieces[temp_id].x, self.__pieces[temp_id].y]           # piece position
        obst_pos = np.array([[item.x, item.y] for item in self.__obst]).flatten()   # obstacle positions
        state = pieces_pos + list(obst_pos)
        return state

    def drawPieces(self, color1, color2):   # draw pieces on board
        for rect in self.__pieces:
            pg.draw.rect(self.__screen, color1, rect)
            
        for rect in self.__obst:
            pg.draw.rect(self.__screen, color2, rect)

    def step(self, piece_id, action):                 # one training step
        done, partial = self.__done()                # return new x and y of pieces

        reward = partial                    # reward for having more pieces
                                            # in the final square
        if done:
            reward += 10
        
        if (action == 0 or action == 3):    # give reward for movement
            reward -= 2                     # negative for moving left and dwon
        else:
            reward += 1                     # positive for moving rihgt and up

        self.move(piece_id, action)          # move pieces
        state = self.__get_state()          # get new state
        
        return reward, state, done

    def vars(self):
        return self.__screen, self.__font, self.__background, self.__X, self.__Y     # return variables to main.py

    @property
    def npieces(self):
        return self.__n**2

    @property
    def state_space(self):
        return self.__n*8       # Number of possible game states, x and y pos. of player and opponent

    @property
    def action_space(self):
        return 4*(self.npieces)  # Action space, 4 moves per piece
