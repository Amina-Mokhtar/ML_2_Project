from numpy import random
from colors import Colors
import pygame as pg
import numpy as np
import itertools

class Env(object):
    def __init__(self, width, height, dim):
        self.__screen = pg.display.set_mode((width, height))
        self.__length = 30
        self.__step = 75
        self.__dim = dim  
        self.__n = dim - 4
        self.__color = itertools.cycle((Colors.WHITE, Colors.BLACK))
        self.__board_length = self.__dim * self.__step
        self.__X = (width - self.__board_length) / 2
        self.__Y = (height - self.__board_length) / 2
        self.__pieces, self.__obst = self.__createPieces()
        self.__background = self.__drawBoard()

    def __drawBoard(self):
        background = pg.Surface((self.__board_length, self.__board_length))
        for y in range(0, self.__board_length, self.__step):
            for x in range(0, self.__board_length, self.__step):
                rect = (x, y, self.__step, self.__step)
                pg.draw.rect(background, next(self.__color), rect)
            next(self.__color)
        return background

    def __pos2coord(self, row, col):
        x = (self.__step / 2) + row * self.__step + self.__X - self.__length / 2
        y = (self.__step / 2) + col * self.__step + self.__Y - self.__length / 2
        return x, y

    def __createPieces(self):
        rects1 = []
        rects2 = []
        if self.__dim - self.__n == 4:
            c_range = range(self.__dim - self.__n, self.__dim) 
            r_range = range(0, self.__n) 

            for c in c_range:
                for r in r_range:
                    x, y = self.__pos2coord(r, c)
                    rect = pg.rect.Rect(x, y, self.__length, self.__length)
                    rects1.append(rect)

            k = 0
            mem = []
            while k < 2*self.__n:
                r = random.randint(self.__dim - 1)
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
        self.__pieces, self.__obst = self.__createPieces()
        state = self.__get_state()
        return state

    def __done(self):
        temp_id = 1

        x = self.__pieces[temp_id].x
        y = self.__pieces[temp_id].y

        return (x >= self.__step*(self.__dim-self.__n) + self.__X and y <= self.__step*self.__n + self.__Y)

    def move(self, id, action):
        if (id < 0 or id > (2*self.__n - 1)):
            return 
        
        rest_pieces = [item for i, item in enumerate(self.__pieces) if i not in [id]] + self.__obst

        if action == 0:
            for item in rest_pieces:
                if (item.x == self.__pieces[id].x - self.__step and item.y == self.__pieces[id].y or 
                    self.__pieces[id].x - self.__step < self.__X):
                    return 
            self.__pieces[id].x -= self.__step 
        elif action == 1:
            for item in rest_pieces:
                if (item.x == self.__pieces[id].x + self.__step and item.y == self.__pieces[id].y or 
                    self.__pieces[id].x + self.__step > self.__board_length + self.__X):
                    return 
            self.__pieces[id].x += self.__step
        elif action == 2:
            for item in rest_pieces:
                if (item.y == self.__pieces[id].y - self.__step and item.x == self.__pieces[id].x or
                    self.__pieces[id].y - self.__step < self.__Y):
                    return 
            self.__pieces[id].y -= self.__step
        elif action == 3:
            for item in rest_pieces:
                if (item.y == self.__pieces[id].y + self.__step and item.x == self.__pieces[id].x or
                    self.__pieces[id].y + self.__step > self.__board_length + self.__Y):
                    return 
            self.__pieces[id].y += self.__step

    def __get_state(self):
        temp_id = 1
        pieces_pos = [self.__pieces[temp_id].x, self.__pieces[temp_id].y]
        obst_pos = np.array([[item.x, item.y] for item in self.__obst]).flatten()
        state = pieces_pos + list(obst_pos)
        return state

    def drawPieces(self, color1, color2):
        for rect in self.__pieces:
            pg.draw.rect(self.__screen, color1, rect)
            
        for rect in self.__obst:
            pg.draw.rect(self.__screen, color2, rect)

    def step(self, action):
        temp_id = 1
        reward = 0

        if (action == 0 or action == 3):
            reward -= 4
        else:
            reward += 4

        self.move(temp_id, action)
        state = self.__get_state()
        done = self.__done()
        return reward, state, done

    def vars(self):
        return self.__screen, self.__background, self.__X, self.__Y

    @property
    def state_space(self):
        return self.__n*8

    @property
    def action_space(self):
        return 4
