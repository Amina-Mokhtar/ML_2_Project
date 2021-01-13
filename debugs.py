import matplotlib.pyplot as plt
import numpy as np

class Debug:
	def __init__(self):
		self.npieces = 4
		self.dim = 4
		self.px = 50

	def save_array(self, array,text):
		for i in range(self.npieces):
			plt.imsave(text+str(i)+".png",(array.reshape((self.dim,self.dim,self.npieces))[:,:,i]).repeat(self.px,0).repeat(self.px,1))
		return
		  
	def save_action(self,action):
		tmp = np.zeros((self.dim,self.dim,self.npieces))
		y, x, piece_id = np.unravel_index(action,(self.dim,self.dim,self.npieces))
		tmp[y,x,piece_id] = 1
		plt.imsave("action"+".png",(tmp.reshape((self.dim,self.dim,self.npieces))[:,:,piece_id]).repeat(self.px,0).repeat(self.px,1))
		return
	
	def save_all(self,pc=None,ob=None,valid_moves=None,actions=None,actions_valid=None,action=0,fname='all'):
		if pc is None:
			pc = np.zeros((self.dim,2))
		if ob is None:
			ob = np.zeros((self.dim,2))
		if valid_moves is None:
			valid_moves = np.zeros((self.dim,self.dim,self.npieces))
		if actions is None:
			actions = np.zeros((self.dim,self.dim,self.npieces))
		if actions_valid is None:
			actions_valid = np.zeros((self.dim,self.dim,self.npieces))
		
		tmp = np.zeros((self.dim,self.dim,self.npieces))
		y, x, piece_id = np.unravel_index(action,(self.dim,self.dim,self.npieces))
		tmp[y,x,piece_id] = 1
		po = np.zeros((self.dim,self.npieces))
		for i in range(self.npieces):
			x,y = pc[i]
			po[int(y),int(x)] = i+1
			x,y = ob[i]
			po[int(y),int(x)] = 5
		fig = plt.figure()
		plt.suptitle('valid moves ,actions ,valid actions ,action')
		for i in range(self.npieces):
				plt.subplot(self.npieces,5,i*5+1)
				plt.imshow(po.repeat(self.px,0).repeat(self.px,1))
				plt.annotate('x',pc[i]*self.px+np.array([self.px/4,self.px/2]))

				plt.subplot(self.npieces,5,i*5+1+1)
				plt.imshow((valid_moves.reshape((self.dim,self.dim,self.npieces))[:,:,i]).repeat(self.px,0).repeat(self.px,1))

				plt.subplot(self.npieces,5,i*5+2+1)
				plt.imshow((actions.reshape((self.dim,self.dim,self.npieces))[:,:,i]).repeat(self.px,0).repeat(self.px,1))
				plt.xlabel('max='+str(actions.max()),fontsize='xx-small')

				plt.subplot(self.npieces,5,i*5+3+1)
				plt.imshow((actions_valid.reshape((self.dim,self.dim,self.npieces))[:,:,i]).repeat(self.px,0).repeat(self.px,1))
				plt.xlabel('max='+str(actions_valid.max()),fontsize='xx-small')
								
				plt.subplot(self.npieces,5,i*5+4+1)
				plt.imshow((tmp.reshape((self.dim,self.dim,self.npieces))[:,:,i]).repeat(self.px,0).repeat(self.px,1))
		fig.savefig(fname+".png")
		return