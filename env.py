from numpy import random
from colors import Colors
from text import Text
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
        self.__moves = 0
        self.__color = itertools.cycle((Colors.WHITE, Colors.BLACK))
        self.__board_length = self.__dim * self.__step
        self.__X = (width - self.__board_length) / 2
        self.__Y = (height - self.__board_length) / 2
        self.__pieces, self.__obst = self.__createPieces()
        self.__background = self.__drawBoard()
        self.__epoch_text, self.__epoch_text_rect = Text.createText("Epoch:", 40, 100)
        self.__moves_text, self.__moves_text_rect = Text.createText("Moves:", 40, 120)
        self.__eps_text, self.__eps_text_rect = Text.createText("eps:", 40, 140)
        self.__done_text, self.__done_text_rect = Text.createText("Done:", 40, 160)
        self.__loss_text, self.__loss_text_rect = Text.createText("Loss:", 40, 180)

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
        self.__moves = 0
        state = self.__getState()
        return state

    def __done(self):
        done = True
        for id in range(self.npieces):
            x = self.__pieces[id].x
            y = self.__pieces[id].y
            done &= x >= self.__step * (self.__dim - self.__n) + self.__X and y <= self.__step * self.__n + self.__Y
        return done

    def availableMoves(self, id):
        if id < 0 or id >= self.npieces:
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
            if action == 0: # left
                self.__pieces[id].x -= self.__step
            elif action == 1: # right
                self.__pieces[id].x += self.__step
            elif action == 2: # up
                self.__pieces[id].y -= self.__step
            elif action == 3: # down
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

    def __getState(self):
        state = np.zeros(shape=(2, self.__dim, self.__dim))
        pos = self.__getPositions()

        for id, loc in enumerate(pos[0:self.__n**2]): # pieces
            state[0][loc[0], loc[1]] = id + 1

        for loc in pos[self.__n**2:len(pos)]: # obstacles
            state[1][loc[0], loc[1]] = 1

        return state

    def getIDMask(self):
        mask = np.zeros(shape=(self.npieces, self.__dim, self.__dim))
        pos = self.__getPositions()

        for id, loc in enumerate(pos[0:self.__n**2]):
            mask[id][loc[0], loc[1]] = id + 1

        return mask

    def mask2num(self, mask):
        b = np.nonzero(mask)
        return int((mask[b].squeeze()))

    def __getReward(self, id, old_pos, new_pos):
        if self.__done():
            return 100
        if new_pos[id] == old_pos[id]:
            return -10
        if (not (old_pos[id][0] in [0, 1] and old_pos[id][1] in [4, 5])) and \
            new_pos[id][0] in [0, 1] and new_pos[id][1] in [4, 5]:
            return -9
        if old_pos[id][0] in [0, 1] and old_pos[id][1] in [4, 5] and \
           new_pos[id][0] in [0, 1] and new_pos[id][1] in [4, 5]:
            return -10
        if old_pos[id][0] in [0, 1] and old_pos[id][1] in [4, 5] and \
           not (new_pos[id][0] in [0, 1] and new_pos[id][1] in [4, 5]):
            return -10
        if new_pos[id][0] == old_pos[id][0] + 2 or new_pos[id][1] == old_pos[id][1] - 2:
            return -10
        if new_pos[id][0] == old_pos[id][0] - 2 or new_pos[id][1] == old_pos[id][1] + 2:
            return -10
        if new_pos[id][0] == old_pos[id][0] + 1 or new_pos[id][1] == old_pos[id][1] - 1:
            return -10
        if new_pos[id][0] == old_pos[id][0] - 1 or new_pos[id][1] == old_pos[id][1] + 1:
            return -10
        return -10

    def __drawPieces(self, color1, color2):
        for rect in self.__pieces:
            pg.draw.rect(self.__screen, color1, rect)

        for rect in self.__obst:
            pg.draw.rect(self.__screen, color2, rect)

    def step(self, action, id):
        old_pos = self.__getPositions()

        self.move(id, action)
        self.__moves += 1

        new_pos = self.__getPositions()
        reward = self.__getReward(id, old_pos, new_pos)
        state = self.__getState()
        done = self.__done()

        return reward, state, done

    def updateEnv(self, nepochs, reward, eps, loss, dones):
        text_epochs, text_epochs_rect = Text.createText(str(nepochs), 90, 100)
        text_moves, text_moves_rect = Text.createText(str(self.__moves), 90, 120)
        text_eps, text_eps_rect = Text.createText(str(np.round(eps, 2)), 90, 140)
        text_done, text_done_rect = Text.createText(str(dones), 90, 160)
        text_loss, text_loss_rect = Text.createText(str(loss), 90, 180)

        self.__screen.fill(Colors.BACKGROUND)
        self.__screen.blit(self.__background, (self.__X, self.__Y))
        self.__screen.blit(text_epochs, text_epochs_rect)
        self.__screen.blit(text_moves, text_moves_rect)
        self.__screen.blit(text_eps, text_eps_rect)
        self.__screen.blit(text_done, text_done_rect)
        self.__screen.blit(text_loss, text_loss_rect)
        self.__screen.blit(self.__epoch_text, self.__epoch_text_rect)
        self.__screen.blit(self.__moves_text, self.__moves_text_rect)
        self.__screen.blit(self.__eps_text, self.__eps_text_rect)
        self.__screen.blit(self.__done_text, self.__done_text_rect)
        self.__screen.blit(self.__loss_text, self.__loss_text_rect)
        self.__drawPieces(Colors.BLUE, Colors.RED)
        pg.display.update()

    @property
    def state_space(self): # needs attention
        return (3, self.__dim, self.__dim)

    @property
    def npieces(self):
        return self.__n ** 2

    @property
    def action_space(self):
        return 4
