from collections import namedtuple
import random
# the structure of memory experience is [states,action, next_states, rewards, max id],
# the max id means that it store the farthest agent id of current agent.
Experience = namedtuple('Experience',
						('states','actions','next_states','rewards','max_id'))

class ReplayMemory:
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Experience(*args)
		self.position = (self.position + 1)%self.capacity
		
	def sample(self,batch_size):
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)
	def get_item(self, position, num_get):
		return self.memory[position-num_get:position]