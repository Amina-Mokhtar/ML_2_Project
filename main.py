import pygame as pg
import matplotlib.pyplot as plt
import time
# from colors import Colors
from env import Env
from agent import Agent
from board import Board

ep = 50                            	# number of episodes
max_moves = 2000							# max number of moves
dim = 6
model_type = 'conv'
non_valid = False

env = Env(dim)                        # create an environment object
board = Board(env,width=800, height=600,draw=0)    # create board
agent = Agent(env, board,model_type,non_valid)			# create an agent

loss, win, eps = agent.train(ep,max_moves)             # Train the agent

note = 'board size:'+str(dim)+', reward:[-distance,win:1000], model:'+model_type+', allow_non_valid:'+str(non_valid)
# note = ''+'Model: '+model_type+'Valid: '+str(non_valid)           # note to add to image
# plt.plot([i for i in range(ep)], loss)
# plt.xlabel('episode')
# plt.ylabel('reward')
# plt.title('Training with ' + str(ep) + ' episodes.\n' + note)
# plt.savefig("training_" + time.strftime("%Y-%m-%d_%H-%M-%S") +".png")

fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)
par1 = host.twinx()
par2 = host.twinx()
par2.spines["right"].set_position(("axes", 1.2))
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

make_patch_spines_invisible(par2)
par2.spines["right"].set_visible(True)
p1, = host.plot([i for i in range(ep)], loss, "b-", label="Loss")
p2, = par1.plot([i for i in range(ep)], eps, "r-", label="Epsilon")
p3, = par2.plot([i for i in range(ep)], win, "g-", label="Done")
host.set_xlabel("Episode")
host.set_ylabel("Loss")
par1.set_ylabel("Epsilon")
par2.set_ylabel("Done")
host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)
lines = [p1, p2, p3]
host.legend(lines, [l.get_label() for l in lines])
host.set_title('Training with ' + str(ep) + ' episodes.\n' + note)
fig.savefig("training_" + time.strftime("%Y-%m-%d_%H-%M-%S") +".png")

agent.save_model(time.strftime("%Y-%m-%d_%H-%M-%S"))