import matplotlib.pyplot as plt
import numpy as np


class Debug:
	def __init__(self):
		 pass

	def save_array(self, array,text):
		for i in range(4):
			plt.imsave(text+str(i)+".png",(array.reshape((6,6,4))[:,:,i]).repeat(50,0).repeat(50,1))
		return
		  
	def save_action(self,action):
		tmp = np.zeros((6,6,4))
		y, x, piece_id = np.unravel_index(action,(6,6,4))
		tmp[y,x,piece_id] = 1
		plt.imsave("action"+".png",(tmp.reshape((6,6,4))[:,:,piece_id]).repeat(50,0).repeat(50,1))
		return
	
	def save_all(self,valid_moves=np.zeros((6,6,4)),actions=np.zeros((6,6,4)),actions_valid=np.zeros((6,6,4)),action=0):
		tmp = np.zeros((6,6,4))
		y, x, piece_id = np.unravel_index(action,(6,6,4))
		tmp[y,x,piece_id] = 1
		# fig = plt.subplots(4,4)
		fig = plt.figure()
		plt.suptitle('valid moves ,actions ,valid actions ,action')
		for i in range(4):
				plt.subplot(4,4,i*4+1)
				plt.imshow((valid_moves.reshape((6,6,4))[:,:,i]).repeat(50,0).repeat(50,1))
				plt.subplot(4,4,i*4+2)

				plt.imshow((actions.reshape((6,6,4))[:,:,i]).repeat(50,0).repeat(50,1))
				plt.xlabel('max='+str(actions.max()),fontsize='xx-small')

				plt.subplot(4,4,i*4+3)
				plt.imshow((actions_valid.reshape((6,6,4))[:,:,i]).repeat(50,0).repeat(50,1))
				plt.xlabel('max='+str(actions_valid.max()),fontsize='xx-small')
								
				plt.subplot(4,4,i*4+4)
				plt.imshow((tmp.reshape((6,6,4))[:,:,i]).repeat(50,0).repeat(50,1))
		fig.savefig("all.png")
		return