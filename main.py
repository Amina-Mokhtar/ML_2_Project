import pygame as pg
from colors import Colors
from env import Env
from agent import Agent
import matplotlib.pyplot as plt
import time

# pg.init()

env = Env(width=800, height=600, dim=6) # create an environment object
agent = Agent(env)                      # create an agent

ep = 2
draw = True
loss = agent.train(ep,draw)                  # Train the agent
plt.plot([i for i in range(ep)], loss)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Training with %i episodes' %ep)
plt.savefig("training_" + time.strftime("%Y-%m-%d_%H-%M-%S") +".png")

screen, background, X, Y = env.vars()   # Get variables from environment

game_exit = False                       # start a game
while not game_exit:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            game_exit = True
        # for i in agent.__memory
        # if event.type == pg.KEYDOWN:
        #     if event.key == pg.K_LEFT:
        #         env.move(1, 0)
        #     if event.key == pg.K_RIGHT:
        #         env.move(1, 1)
        #     if event.key == pg.K_UP:
        #         env.move(1, 2)
        #     if event.key == pg.K_DOWN:
        #         env.move(1, 3)

    if not draw:             
        screen.fill(Colors.BACKGROUND)  # fill screen with background colour
        screen.blit(background, (X, Y)) # draw board squares onto screen
        env.drawPieces(Colors.BLUE, Colors.RED) # draw pieces on board
        pg.display.update()

pg.quit()

