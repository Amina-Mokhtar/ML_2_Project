import pygame as pg
from env import Env
from agent import Agent
import matplotlib.pyplot as plt

pg.init()
pg.display.set_caption("Corners")

env = Env(width=800, height=600, dim=6)
agent = Agent(env)

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

    loss, move = agent.train(100)
    game_exit = True
pg.quit()

plt.plot(loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot([m[1] for m in move], [e[0] for e in move])
plt.xlabel("Epoch")
plt.ylabel("Move")
plt.show()