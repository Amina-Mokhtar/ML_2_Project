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

    def __getPositions(self):
        pos = []
        for item in self.__pieces + self.__obst:
            x = int(np.floor((item.x - self.__X) / self.__step))
            y = int(np.floor((item.y - self.__Y) / self.__step))
            pos.append([y, x])
        return pos

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
            while k < 2 * self.__n:
                r = random.randint(self.__dim - 1)
                c = random.randint(self.__dim - 1)
                if not ((c in c_range and r in r_range) or (c in r_range and r in c_range)):
                    if (r, c) not in mem:
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
        done = True
        for id in range(self.__n**2):
            x = self.__pieces[id].x
            y = self.__pieces[id].y
            done &= x >= self.__step * (self.__dim - self.__n) + self.__X and y <= self.__step * self.__n + self.__Y
        return done

    def availableMoves(self, id):
        if id < 0 or id > (2 * self.__n - 1):
            return

        rest_pieces = [item for i, item in enumerate(self.__pieces) if i not in [id]] + self.__obst
        moves = []
        jumps = []

        for action in range(self.action_space):
            if action == 0: # left
                jump = False
                if self.__pieces[id].x - self.__step < self.__X:
                    continue
                for item in rest_pieces:
                    if item.x == self.__pieces[id].x - self.__step and item.y == self.__pieces[id].y:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.__pieces[id].x - 2 * self.__step < self.__X:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.x == self.__pieces[id].x - 2 * self.__step and item.y == self.__pieces[id].y:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)

            elif action == 1: # right
                jump = False
                if self.__pieces[id].x + self.__step > self.__board_length + self.__X:
                    continue
                for item in rest_pieces:
                    if item.x == self.__pieces[id].x + self.__step and item.y == self.__pieces[id].y:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.__pieces[id].x + 2 * self.__step > self.__board_length + self.__X:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.x == self.__pieces[id].x + 2 * self.__step and item.y == self.__pieces[id].y:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)

            elif action == 2: # up
                jump = False
                if self.__pieces[id].y - self.__step < self.__Y:
                    continue
                for item in rest_pieces:
                    if item.y == self.__pieces[id].y - self.__step and item.x == self.__pieces[id].x:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.__pieces[id].y - 2 * self.__step < self.__Y:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.y == self.__pieces[id].y - 2 * self.__step and item.x == self.__pieces[id].x:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)

            elif action == 3: # down
                jump = False
                if self.__pieces[id].y + self.__step > self.__board_length + self.__Y:
                    continue
                for item in rest_pieces:
                    if item.y == self.__pieces[id].y + self.__step and item.x == self.__pieces[id].x:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.__pieces[id].y + 2 * self.__step > self.__board_length + self.__Y:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.y == self.__pieces[id].y + 2 * self.__step and item.x == self.__pieces[id].x:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)
        return moves, jumps

    def move(self, id, action):
        moves, jumps = self.availableMoves(id)

        if action in moves:
            if action == 0:  # left
                self.__pieces[id].x -= self.__step
            elif action == 1:  # right
                self.__pieces[id].x += self.__step
            elif action == 2:  # up
                self.__pieces[id].y -= self.__step
            elif action == 3:  # down
                self.__pieces[id].y += self.__step
        elif action in jumps:
            if action == 0:
                self.__pieces[id].x -= 2 * self.__step
            elif action == 1:
                self.__pieces[id].x += 2 * self.__step
            elif action == 2:
                self.__pieces[id].y -= 2 * self.__step
            elif action == 3:
                self.__pieces[id].y += 2 * self.__step

    def __get_state(self):
        state = np.zeros(shape=(self.__dim, self.__dim))
        pos = self.__getPositions()

        for loc in pos[0:self.__n**2]:
            state[loc[0], loc[1]] = 1

        for loc in pos[self.__n**2:len(pos)]:
            state[loc[0], loc[1]] = 2

        return state.flatten()

    def __getRewardMatrix(self):
        A = np.zeros(shape=(self.__dim, self.__dim))
        pos = self.__getPositions()

        for r in range(1, self.__dim + 1):
            for c in range(1, self.__dim + 1):
                A[self.__dim - c, r - 1] = r * c

        for i in range(len(pos)):
            if i != 1: # fix me boy
                A[pos[i][0], pos[i][1]] = 0
        return A

    def drawPieces(self, color1, color2):
        for rect in self.__pieces:
            pg.draw.rect(self.__screen, color1, rect)

        for rect in self.__obst:
            pg.draw.rect(self.__screen, color2, rect)

    def step(self, action, id):
        reward_mat = self.__getRewardMatrix()

        old_pos = self.__getPositions()

        self.move(id, action)

        new_pos = self.__getPositions()

        if new_pos == old_pos:
            reward = -10
        else:
            reward = reward_mat[new_pos[id][0], new_pos[id][1]] - reward_mat[
                old_pos[id][0], old_pos[id][1]]

        done = self.__done()
        state = self.__get_state()
        return reward, state, done

    def vars(self):
        return self.__screen, self.__background, self.__X, self.__Y

    @property
    def state_space(self): # needs attention
        return self.__n * 8

    @property
    def action_space(self):
        return 4
