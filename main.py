import pygame as pg
import matplotlib.pyplot as plt
import time
# from colors import Colors
from env import Env
from agent import Agent
from board import Board

ep = 200                            	# number of episodes
max_moves = 500								# max number of moves

env = Env(dim=4)                        # create an environment object
board = Board(env,width=800, height=600,draw=0)    # create board
agent = Agent(env, board,model_type='conv',non_valid=False)			# create an agent

loss = agent.train(ep,max_moves)             # Train the agent

note = 'reward: partial/4,done=10'           # note to add to image
plt.plot([i for i in range(ep)], loss)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Training with ' + str(ep) + ' episodes.\n' + note)
plt.savefig("training_" + time.strftime("%Y-%m-%d_%H-%M-%S") +".png")

agent.save_model(time.strftime("%Y-%m-%d_%H-%M-%S"))