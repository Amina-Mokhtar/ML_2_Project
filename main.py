
import pygame as pg
from colors import Colors
from env import Env
from agent import Agent


pg.init()

env = Env(width=800, height=600, dim=6)

agent = Agent(env)
loss = agent.train(50)

screen, background, X, Y = env.vars()

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

    screen.fill(Colors.BACKGROUND)
    screen.blit(background, (X, Y))

    env.drawPieces(Colors.BLUE, Colors.RED)

    pg.display.update()

pg.quit()