import pygame as pg
import matplotlib.pyplot as plt
import time
from colors import Colors
from env import Env
from agent import Agent
from board import Board

ep = 1000                            	# number of episodes
max_moves = 150								# max number of moves

env = Env(dim=6)                        # create an environment object
board = Board(env,width=800, height=600,draw=-1)    # create board
agent = Agent(env, board)                      # create an agent

loss = agent.train(ep,max_moves)             # Train the agent

note = ''                               # note to add to image
plt.plot([i for i in range(ep)], loss)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Training with ' + str(ep) + ' episodes.' + note)
plt.savefig("training_" + time.strftime("%Y-%m-%d_%H-%M-%S") +".png")