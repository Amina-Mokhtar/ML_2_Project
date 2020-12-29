from numpy import random
from colors import Colors
import pygame as pg
import numpy as np
import itertools
from env import Env
from colors import Colors
import time

class Board(object):
   '''
   Draw the game using pygame
   '''
   def __init__(self, env, width, height):
      pg.init()
      self.__env = env
      self.__screen = pg.display.set_mode((width, height))    # pygame screen object
      self.__font = pg.font.SysFont(None, 24)
      self.__length = 30                                      # size of pieces
      self.__step = 75                                        # width of board squares (px)
      self.__dim = self.__env.dim
      self.__board_length = self.__dim * self.__step          # board size in px
      self.__X = (width - self.__board_length) / 2            # x coordinate of top-left corner of board
      self.__Y = (height - self.__board_length) / 2           # y coordinate of top-left corner of board
      self.__color = itertools.cycle((Colors.WHITE, Colors.BLACK))    # board colours
      self.__background = self.__drawBG()                  # board object, draws the board

   def __drawBG(self):
      background = pg.Surface((self.__board_length, self.__board_length)) # pygame surface of board
      for y in range(0, self.__board_length, self.__step):
         for x in range(0, self.__board_length, self.__step):
            rect = (x, y, self.__step, self.__step)                     
            pg.draw.rect(background, next(self.__color), rect)          # draw rectangles to create board pattern
         next(self.__color)
      return background

   def draw_board(self,text=''):
      self.__screen.fill(Colors.BACKGROUND)  # fill screen with background colour
      self.__screen.blit(self.__background, (self.__X, self.__Y)) # draw board squares onto screen
      img = self.__font.render(text, True, Colors.WHITE)
      self.__screen.blit(img, (20, 20))
      self.drawPieces() # draw pieces on board
      pg.display.update()
      for event in pg.event.get():
         if event.type == pg.QUIT:
            pg.quit()
      time.sleep(0.5)

   def __pos2coord(self, row, col):    # convert row/column to coordinates in window
      x = (self.__step / 2) + row * self.__step + self.__X - self.__length / 2
      y = (self.__step / 2) + col * self.__step + self.__Y - self.__length / 2
      return x, y

   def drawPieces(self):   # draw pieces on board from piece and obst vectors
      pieces, obst = self.__env.return_pieces()
      for i in range(self.__env.npieces):
         r,c = pieces[i]
         x,y = self.__pos2coord(r, c)
         rect = pg.rect.Rect(x,y,self.__length,self.__length)
         pg.draw.rect(self.__screen, Colors.BLUE, rect)
         img = self.__font.render(str(i), True, Colors.WHITE)
         self.__screen.blit(img, (x, y))

         r,c = obst[i]
         x,y = x, y = self.__pos2coord(r, c)
         rect = pg.rect.Rect(x,y,self.__length,self.__length)
         pg.draw.rect(self.__screen, Colors.RED, rect)
         img = self.__font.render(str(i), True, Colors.WHITE)
         self.__screen.blit(img, (x, y))
