# from numpy import random
# from colors import Colors
# from text import Text
# import pygame as pg
# import numpy as np
# import itertools
#
#
# class Env(object):
#     def __init__(self, width, height, dim):
#         self.__screen = pg.display.set_mode((width, height))
#         self.__length = 30
#         self.__step = 75
#         self.__dim = dim
#         self.__n = dim - 4
#         self.__moves = 0
#         self.__color = itertools.cycle((Colors.WHITE, Colors.BLACK))
#         self.__board_length = self.__dim * self.__step
#         self.__X = (width - self.__board_length) / 2
#         self.__Y = (height - self.__board_length) / 2
#         self.__pieces, self.__obst = self.__createPieces()
#         self.__background = self.__drawBoard()
#         self.__epoch_text, self.__epoch_text_rect = Text.createText("Epoch:", 40, 100)
#         self.__moves_text, self.__moves_text_rect = Text.createText("Moves:", 40, 120)
#         self.__eps_text, self.__eps_text_rect = Text.createText("eps:", 40, 140)
#         self.__done_text, self.__done_text_rect = Text.createText("Done:", 40, 160)
#         self.__loss_text, self.__loss_text_rect = Text.createText("Loss:", 40, 180)
#
#     def __drawBoard(self):
#         background = pg.Surface((self.__board_length, self.__board_length))
#         for y in range(0, self.__board_length, self.__step):
#             for x in range(0, self.__board_length, self.__step):
#                 rect = (x, y, self.__step, self.__step)
#                 pg.draw.rect(background, next(self.__color), rect)
#             next(self.__color)
#         return background
#
#     def __pos2coord(self, row, col):
#         x = (self.__step / 2) + row * self.__step + self.__X - self.__length / 2
#         y = (self.__step / 2) + col * self.__step + self.__Y - self.__length / 2
#         return x, y
#
#     def __getPositions(self):
#         pos = []
#         for item in self.__pieces + self.__obst:
#             x = int(np.floor((item.x - self.__X) / self.__step))
#             y = int(np.floor((item.y - self.__Y) / self.__step))
#             pos.append([y, x])
#         return pos
#
#     def __createPieces(self):
#         rects1 = []
#         rects2 = []
#         if self.__dim - self.__n == 4:
#             c_range = range(self.__dim - self.__n, self.__dim)
#             r_range = range(0, self.__n)
#
#             for c in c_range:
#                 for r in r_range:
#                     x, y = self.__pos2coord(r, c)
#                     rect = pg.rect.Rect(x, y, self.__length, self.__length)
#                     rects1.append(rect)
#
#             k = 0
#             mem = []
#             while k < 2 * self.__n:
#                 r = random.randint(self.__dim - 1)
#                 c = random.randint(self.__dim - 1)
#                 if not ((c in c_range and r in r_range) or (c in r_range and r in c_range)):
#                     if (r, c) not in mem:
#                         mem.append((r, c))
#                         x, y = self.__pos2coord(r, c)
#                         rect = pg.rect.Rect(x, y, self.__length, self.__length)
#                         rects2.append(rect)
#                         k += 1
#
#         return rects1, rects2
#
#     def reset(self):
#         self.__pieces, self.__obst = self.__createPieces()
#         self.__moves = 0
#         state = self.__getState()
#         return state
#
#     def __done(self):
#         done = True
#         for id in range(self.npieces):
#             x = self.__pieces[id].x
#             y = self.__pieces[id].y
#             done &= x >= self.__step * (self.__dim - self.__n) + self.__X and y <= self.__step * self.__n + self.__Y
#         return done
#
#     def availableMoves(self, id):
#         if id < 0 or id >= self.npieces:
#             return
#
#         rest_pieces = [item for i, item in enumerate(self.__pieces) if i not in [id]] + self.__obst
#         moves = []
#         jumps = []
#
#         for action in range(self.action_space):
#             if action == 0: # left
#                 jump = False
#                 if self.__pieces[id].x - self.__step < self.__X:
#                     continue
#                 for item in rest_pieces:
#                     if item.x == self.__pieces[id].x - self.__step and item.y == self.__pieces[id].y:
#                         jump = True
#                 if not jump:
#                     moves.append(action)
#                 else:
#                     if self.__pieces[id].x - 2 * self.__step < self.__X:
#                         continue
#                     flag = False
#                     for item in rest_pieces:
#                         if item.x == self.__pieces[id].x - 2 * self.__step and item.y == self.__pieces[id].y:
#                             flag = True
#                     if flag:
#                         continue
#                     jumps.append(action)
#
#             elif action == 1: # right
#                 jump = False
#                 if self.__pieces[id].x + self.__step > self.__board_length + self.__X:
#                     continue
#                 for item in rest_pieces:
#                     if item.x == self.__pieces[id].x + self.__step and item.y == self.__pieces[id].y:
#                         jump = True
#                 if not jump:
#                     moves.append(action)
#                 else:
#                     if self.__pieces[id].x + 2 * self.__step > self.__board_length + self.__X:
#                         continue
#                     flag = False
#                     for item in rest_pieces:
#                         if item.x == self.__pieces[id].x + 2 * self.__step and item.y == self.__pieces[id].y:
#                             flag = True
#                     if flag:
#                         continue
#                     jumps.append(action)
#
#             elif action == 2: # up
#                 jump = False
#                 if self.__pieces[id].y - self.__step < self.__Y:
#                     continue
#                 for item in rest_pieces:
#                     if item.y == self.__pieces[id].y - self.__step and item.x == self.__pieces[id].x:
#                         jump = True
#                 if not jump:
#                     moves.append(action)
#                 else:
#                     if self.__pieces[id].y - 2 * self.__step < self.__Y:
#                         continue
#                     flag = False
#                     for item in rest_pieces:
#                         if item.y == self.__pieces[id].y - 2 * self.__step and item.x == self.__pieces[id].x:
#                             flag = True
#                     if flag:
#                         continue
#                     jumps.append(action)
#
#             elif action == 3: # down
#                 jump = False
#                 if self.__pieces[id].y + self.__step > self.__board_length + self.__Y:
#                     continue
#                 for item in rest_pieces:
#                     if item.y == self.__pieces[id].y + self.__step and item.x == self.__pieces[id].x:
#                         jump = True
#                 if not jump:
#                     moves.append(action)
#                 else:
#                     if self.__pieces[id].y + 2 * self.__step > self.__board_length + self.__Y:
#                         continue
#                     flag = False
#                     for item in rest_pieces:
#                         if item.y == self.__pieces[id].y + 2 * self.__step and item.x == self.__pieces[id].x:
#                             flag = True
#                     if flag:
#                         continue
#                     jumps.append(action)
#         return moves, jumps
#
#     def move(self, id, action):
#         moves, jumps = self.availableMoves(id)
#
#         if action in moves:
#             if action == 0: # left
#                 self.__pieces[id].x -= self.__step
#             elif action == 1: # right
#                 self.__pieces[id].x += self.__step
#             elif action == 2: # up
#                 self.__pieces[id].y -= self.__step
#             elif action == 3: # down
#                 self.__pieces[id].y += self.__step
#         elif action in jumps:
#             if action == 0:
#                 self.__pieces[id].x -= 2 * self.__step
#             elif action == 1:
#                 self.__pieces[id].x += 2 * self.__step
#             elif action == 2:
#                 self.__pieces[id].y -= 2 * self.__step
#             elif action == 3:
#                 self.__pieces[id].y += 2 * self.__step
#
#     def __getState(self):
#         state = np.zeros(shape=(2, self.__dim, self.__dim))
#         pos = self.__getPositions()
#
#         for id, loc in enumerate(pos[0:self.__n**2]): # pieces
#             state[0][loc[0], loc[1]] = id + 1
#
#         for loc in pos[self.__n**2:len(pos)]: # obstacles
#             state[1][loc[0], loc[1]] = 1
#
#         return state
#
#     def getIDMask(self):
#         mask = np.zeros(shape=(self.npieces, self.__dim, self.__dim))
#         pos = self.__getPositions()
#
#         for id, loc in enumerate(pos[0:self.__n**2]):
#             mask[id][loc[0], loc[1]] = id + 1
#
#         return mask
#
#     def mask2num(self, mask):
#         b = np.nonzero(mask)
#         return int((mask[b].squeeze()))
#
#     def __getReward(self, id, old_pos, new_pos):
#         if self.__done():
#             return 100
#         if new_pos[id] == old_pos[id]:
#             return -10
#         if (not (old_pos[id][0] in [0, 1] and old_pos[id][1] in [4, 5])) and \
#             new_pos[id][0] in [0, 1] and new_pos[id][1] in [4, 5]:
#             return -9
#         if old_pos[id][0] in [0, 1] and old_pos[id][1] in [4, 5] and \
#            new_pos[id][0] in [0, 1] and new_pos[id][1] in [4, 5]:
#             return -10
#         if old_pos[id][0] in [0, 1] and old_pos[id][1] in [4, 5] and \
#            not (new_pos[id][0] in [0, 1] and new_pos[id][1] in [4, 5]):
#             return -10
#         if new_pos[id][0] == old_pos[id][0] + 2 or new_pos[id][1] == old_pos[id][1] - 2:
#             return -10
#         if new_pos[id][0] == old_pos[id][0] - 2 or new_pos[id][1] == old_pos[id][1] + 2:
#             return -10
#         if new_pos[id][0] == old_pos[id][0] + 1 or new_pos[id][1] == old_pos[id][1] - 1:
#             return -10
#         if new_pos[id][0] == old_pos[id][0] - 1 or new_pos[id][1] == old_pos[id][1] + 1:
#             return -10
#         return -10
#
#     def __drawPieces(self, color1, color2):
#         for rect in self.__pieces:
#             pg.draw.rect(self.__screen, color1, rect)
#
#         for rect in self.__obst:
#             pg.draw.rect(self.__screen, color2, rect)
#
#     def step(self, action, id):
#         old_pos = self.__getPositions()
#
#         self.move(id, action)
#         self.__moves += 1
#
#         new_pos = self.__getPositions()
#         reward = self.__getReward(id, old_pos, new_pos)
#         state = self.__getState()
#         done = self.__done()
#
#         return reward, state, done
#
#     def updateEnv(self, nepochs, reward, eps, loss, dones):
#         text_epochs, text_epochs_rect = Text.createText(str(nepochs), 90, 100)
#         text_moves, text_moves_rect = Text.createText(str(self.__moves), 90, 120)
#         text_eps, text_eps_rect = Text.createText(str(np.round(eps, 2)), 90, 140)
#         text_done, text_done_rect = Text.createText(str(dones), 90, 160)
#         text_loss, text_loss_rect = Text.createText(str(loss), 90, 180)
#
#         self.__screen.fill(Colors.BACKGROUND)
#         self.__screen.blit(self.__background, (self.__X, self.__Y))
#         self.__screen.blit(text_epochs, text_epochs_rect)
#         self.__screen.blit(text_moves, text_moves_rect)
#         self.__screen.blit(text_eps, text_eps_rect)
#         self.__screen.blit(text_done, text_done_rect)
#         self.__screen.blit(text_loss, text_loss_rect)
#         self.__screen.blit(self.__epoch_text, self.__epoch_text_rect)
#         self.__screen.blit(self.__moves_text, self.__moves_text_rect)
#         self.__screen.blit(self.__eps_text, self.__eps_text_rect)
#         self.__screen.blit(self.__done_text, self.__done_text_rect)
#         self.__screen.blit(self.__loss_text, self.__loss_text_rect)
#         self.__drawPieces(Colors.BLUE, Colors.RED)
#         pg.display.update()
#
#     @property
#     def state_space(self): # needs attention
#         return (3, self.__dim, self.__dim)
#
#     @property
#     def npieces(self):
#         return self.__n ** 2
#
#     @property
#     def action_space(self):
#         return 4

from numpy import random
from colors import Colors
from text import Text
import pygame as pg
import numpy as np
import itertools

class Env(object):
    def __init__(self, width, height, dim):
        self.screen = pg.display.set_mode((width, height))
        self.length = 30
        self.step = 75
        self.dim = dim
        self.n = dim - 4
        self.moves = 0
        self.colors = itertools.cycle((Colors.WHITE, Colors.BLACK))
        self.board_length = self.dim * self.step
        self.X = (width - self.board_length) / 2
        self.Y = (height - self.board_length) / 2
        self.pieces, self.obst = self.createPieces()
        self.background = self.drawBoard()
        self.epoch_text, self.epoch_text_rect = Text.createText("Epoch:", 40, 100)
        self.moves_text, self.moves_text_rect = Text.createText("Moves:", 40, 120)
        self.eps_text, self.eps_text_rect = Text.createText("eps:", 40, 140)
        self.done_text, self.done_text_rect = Text.createText("Done:", 40, 160)
        self.loss_text, self.loss_text_rect = Text.createText("Loss:", 40, 180)

    def drawBoard(self):
        background = pg.Surface((self.board_length, self.board_length))
        for y in range(0, self.board_length, self.step):
            for x in range(0, self.board_length, self.step):
                rect = (x, y, self.step, self.step)
                pg.draw.rect(background, next(self.colors), rect)
            next(self.colors)
        return background

    def pos2coord(self, row, col):
        x = (self.step / 2) + row * self.step + self.X - self.length / 2
        y = (self.step / 2) + col * self.step + self.Y - self.length / 2
        return x, y

    def getPositions(self):
        pos = []
        for item in self.pieces + self.obst:
            x = int(np.floor((item.x - self.X) / self.step))
            y = int(np.floor((item.y - self.Y) / self.step))
            pos.append([y, x])
        return pos

    def createPieces(self):
        rects1 = []
        rects2 = []
        if self.dim - self.n == 4:
            c_range = range(self.dim - self.n, self.dim)
            r_range = range(0, self.n)

            for c in c_range:
                for r in r_range:
                    x, y = self.pos2coord(r, c)
                    rect = pg.rect.Rect(x, y, self.length, self.length)
                    rects1.append(rect)

            k = 0
            mem = []
            while k < 2 * self.n:
                r = random.randint(self.dim - 1)
                c = random.randint(self.dim - 1)
                if not ((c in c_range and r in r_range) or (c in r_range and r in c_range)):
                    if (r, c) not in mem:
                        mem.append((r, c))
                        x, y = self.pos2coord(r, c)
                        rect = pg.rect.Rect(x, y, self.length, self.length)
                        rects2.append(rect)
                        k += 1

        return rects1, rects2

    def reset(self):
        self.pieces, self.obst = self.createPieces()
        self.moves = 0
        state = self.getState()
        return state

    def done(self):
        done = True
        for piece_id in range(self.npieces):
            x = self.pieces[piece_id].x
            y = self.pieces[piece_id].y
            done &= x >= self.step * (self.dim - self.n) + self.X and y <= self.step * self.n + self.Y
        return done

    def availableMoves(self, id):
        if id < 0 or id >= self.npieces:
            return

        rest_pieces = [item for i, item in enumerate(self.pieces) if i not in [id]] + self.obst
        moves = []
        jumps = []

        for action in range(self.action_space):
            if action == 0: # left
                jump = False
                if self.pieces[id].x - self.step < self.X:
                    continue
                for item in rest_pieces:
                    if item.x == self.pieces[id].x - self.step and item.y == self.pieces[id].y:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.pieces[id].x - 2 * self.step < self.X:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.x == self.pieces[id].x - 2 * self.step and item.y == self.pieces[id].y:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)

            elif action == 1: # right
                jump = False
                if self.pieces[id].x + self.step > self.board_length + self.X:
                    continue
                for item in rest_pieces:
                    if item.x == self.pieces[id].x + self.step and item.y == self.pieces[id].y:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.pieces[id].x + 2 * self.step > self.board_length + self.X:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.x == self.pieces[id].x + 2 * self.step and item.y == self.pieces[id].y:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)

            elif action == 2: # up
                jump = False
                if self.pieces[id].y - self.step < self.Y:
                    continue
                for item in rest_pieces:
                    if item.y == self.pieces[id].y - self.step and item.x == self.pieces[id].x:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.pieces[id].y - 2 * self.step < self.Y:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.y == self.pieces[id].y - 2 * self.step and item.x == self.pieces[id].x:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)

            elif action == 3: # down
                jump = False
                if self.pieces[id].y + self.step > self.board_length + self.Y:
                    continue
                for item in rest_pieces:
                    if item.y == self.pieces[id].y + self.step and item.x == self.pieces[id].x:
                        jump = True
                if not jump:
                    moves.append(action)
                else:
                    if self.pieces[id].y + 2 * self.step > self.board_length + self.Y:
                        continue
                    flag = False
                    for item in rest_pieces:
                        if item.y == self.pieces[id].y + 2 * self.step and item.x == self.pieces[id].x:
                            flag = True
                    if flag:
                        continue
                    jumps.append(action)
        return moves, jumps

    def move(self, id, action):
        moves, jumps = self.availableMoves(id)

        if action in moves:
            if action == 0: # left
                self.pieces[id].x -= self.step
            elif action == 1: # right
                self.pieces[id].x += self.step
            elif action == 2: # up
                self.pieces[id].y -= self.step
            elif action == 3: # down
                self.pieces[id].y += self.step
        elif action in jumps:
            if action == 0:
                self.pieces[id].x -= 2 * self.step
            elif action == 1:
                self.pieces[id].x += 2 * self.step
            elif action == 2:
                self.pieces[id].y -= 2 * self.step
            elif action == 3:
                self.pieces[id].y += 2 * self.step

    def getState(self):
        state = np.zeros(shape=(2, self.dim, self.dim))
        pos = self.getPositions()

        for id, loc in enumerate(pos[0:self.n**2]): # pieces
            state[0][loc[0], loc[1]] = id + 1

        for loc in pos[self.n**2:len(pos)]: # obstacles
            state[1][loc[0], loc[1]] = 1

        return state

    def getIDMask(self):
        mask = np.zeros(shape=(self.npieces, self.dim, self.dim))
        pos = self.getPositions()

        for id, loc in enumerate(pos[0:self.n**2]):
            mask[id][loc[0], loc[1]] = id + 1

        return mask

    def mask2num(self, mask):
        b = np.nonzero(mask)
        return int((mask[b].squeeze()))

    def getReward(self, id, old_pos, new_pos):
        if self.done():
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
        return -10

    def drawPieces(self, color1, color2):
        for rect in self.pieces:
            pg.draw.rect(self.screen, color1, rect)

        for rect in self.obst:
            pg.draw.rect(self.screen, color2, rect)

    def stepEnv(self, action, id):
        old_pos = self.getPositions()

        self.move(id, action)
        self.moves += 1

        new_pos = self.getPositions()
        reward = self.getReward(id, old_pos, new_pos)
        state = self.getState()
        done = self.done()

        return reward, state, done

    def updateEnv(self, nepochs, reward, eps, loss, dones):
        text_epochs, text_epochs_rect = Text.createText(str(nepochs), 90, 100)
        text_moves, text_moves_rect = Text.createText(str(self.moves), 90, 120)
        text_eps, text_eps_rect = Text.createText(str(np.round(eps, 2)), 90, 140)
        text_done, text_done_rect = Text.createText(str(dones), 90, 160)
        text_loss, text_loss_rect = Text.createText(str(loss), 90, 180)

        self.screen.fill(Colors.BACKGROUND)
        self.screen.blit(self.background, (self.X, self.Y))
        self.screen.blit(text_epochs, text_epochs_rect)
        self.screen.blit(text_moves, text_moves_rect)
        self.screen.blit(text_eps, text_eps_rect)
        self.screen.blit(text_done, text_done_rect)
        self.screen.blit(text_loss, text_loss_rect)
        self.screen.blit(self.epoch_text, self.epoch_text_rect)
        self.screen.blit(self.moves_text, self.moves_text_rect)
        self.screen.blit(self.eps_text, self.eps_text_rect)
        self.screen.blit(self.done_text, self.done_text_rect)
        self.screen.blit(self.loss_text, self.loss_text_rect)
        self.drawPieces(Colors.BLUE, Colors.RED)
        pg.display.update()

    @property
    def state_space(self):
        return (3, self.dim, self.dim)

    @property
    def npieces(self):
        return self.n ** 2

    @property
    def action_space(self):
        return 4
