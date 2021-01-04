import pygame as pg
import matplotlib.pyplot as plt
import time
from colors import Colors
from env import Env
from agent import Agent
from board import Board

ep = 200                              # number of episodes

env = Env(dim=6)                        # create an environment object
board = Board(env,width=800, height=600,draw=0)    # create board
agent = Agent(env, board)                      # create an agent

loss = agent.train(ep)             # Train the agent

note = ''                               # note to add to image
plt.plot([i for i in range(ep)], loss)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Training with ' + str(ep) + ' episodes.' + note)
plt.savefig("training_" + time.strftime("%Y-%m-%d_%H-%M-%S") +".png")