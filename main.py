
import pygame as pg
from colors import Colors
from env import Env
from agent import Agent


pg.init()

env = Env(width=800, height=600, dim=6) # create an environment object

agent = Agent(env)                      # create an agent
loss = agent.train(2)                   # Train the agent # FIXME: Is this correct?

screen, background, X, Y = env.vars()   # Get variables from environment

game_exit = False
while not game_exit:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            game_exit = True
        # if event.type == pg.KEYDOWN:
        #     if event.key == pg.K_LEFT:
        #         env.move(1, 0)
        #     if event.key == pg.K_RIGHT:
        #         env.move(1, 1)
        #     if event.key == pg.K_UP:
        #         env.move(1, 2)
        #     if event.key == pg.K_DOWN:
        #         env.move(1, 3)
                 
    screen.fill(Colors.BACKGROUND)  # fill screen with background colour
    screen.blit(background, (X, Y)) # draw board squares onto screen

    env.drawPieces(Colors.BLUE, Colors.RED) # draw pieces on board
    
    pg.display.update()

pg.quit()

    