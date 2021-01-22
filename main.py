import pygame as pg
from env import Env
from DQN import DQN
from Double_DQN import Double_DQN
from Dueling_DQN import Dueling_DQN
from Noisy_DQN import Noisy_DQN
from PER_DQN import PER_DQN
import matplotlib.pyplot as plt

pg.init()
pg.display.set_caption("Corners")

env = Env(width=800, height=600, dim=6)

agent_DQN = DQN(env)
agent_Double_DQN = Double_DQN(env)
agent_Dueling_DQN = Dueling_DQN(env)
agent_PER_DQN = PER_DQN(env)
agent_Noisy_DQN = Noisy_DQN(env)

labels = ['DQN', 'Double_DQN', 'Dueling_DQN', 'PER_DQN', 'Noisy_DQN']

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

    # loss_1, move_1 = agent_DQN.train(epochs=30, max_moves=2000)
    # loss_2, move_2 = agent_Double_DQN.train(epochs=30, max_moves=2000)
    # loss_3, move_3 = agent_Dueling_DQN.train(epochs=30, max_moves=2000)
    # loss_4, move_4 = agent_PER_DQN.train(epochs=30, max_moves=2000)
    loss_5, move_5 = agent_Noisy_DQN.train(epochs=30, max_moves=2000)
    game_exit = True
pg.quit()

# ls = [loss_1, loss_2, loss_3, loss_4]
ls=[loss_5]
for i in range(len(ls)):
    plt.plot(ls[i], label=labels[i])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(labels, loc="best")
plt.grid()
plt.show()

# mo = [move_1, move_2, move_3, move_4]
mo = [move_5]
for item in mo:
    plt.plot([e[1] for e in item], [m[0] for m in item], label=labels[i])
plt.xlabel("Epoch")
plt.ylabel("Move")
plt.legend(labels, loc="best")
plt.grid()
plt.show()
