import pygame as pg
from env import Env
from Agent_DQN import AgentBaseDeepQNet
from hyperparameters import hyperparameters as hp
from Agent_DDQN import AgentDoubleDeepQNet
from Agent_Dueling_DQN import AgentDuelingDeepQNet
from Agent_Dueling_DDQN import AgentDuelingDoubleDeepQNet
from Agent_PER_DQN import AgentPERDeepQNet
import matplotlib.pyplot as plt

pg.init()
pg.display.set_caption("Corners")

env = Env(width=800, height=600, dim=6)

agents = []
agents.append(AgentBaseDeepQNet(env, hp.lr, hp.gamma, hp.batch_size, hp.eps_decay))
agents.append(AgentDoubleDeepQNet(env, hp.lr, hp.gamma, hp.batch_size, hp.eps_decay))
agents.append(AgentDuelingDeepQNet(env, hp.lr, hp.gamma, hp.batch_size, hp.eps_decay))
agents.append(AgentDuelingDoubleDeepQNet(env, hp.lr, hp.gamma, hp.batch_size, hp.eps_decay))
agents.append(AgentPERDeepQNet(env, hp.lr, hp.gamma, hp.batch_size, hp.eps_decay))

labels = ['DQN', 'DDQN', 'Dueling_DQN', 'Dueling_DDQN', 'PER_DQN']

game_exit = False
losses, moves = [], []
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

    for agent in agents:
        loss, move = agent.play(epochs=50, max_moves=2000)
        losses.append(loss)
        moves.append(move)

    game_exit = True
pg.quit()


for i in range(len(losses)):
    plt.plot(losses[i], label=labels[i])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(labels, loc="best")
plt.grid()
plt.show()


for i in range(len(moves)):
    plt.plot([e[1] for e in moves[i]], [m[0] for m in moves[i]], label=labels[i])
plt.xlabel("Epoch")
plt.ylabel("Move")
plt.legend(labels, loc="best")
plt.grid()
plt.show()
