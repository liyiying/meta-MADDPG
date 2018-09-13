###############################################################################
'''
file name: meta_actor.py
function: Train a generic agent metat actor model with meta properties based on the four agents that have been trained on the premise.
           Here, this code has not contained the rnn structure.
note: you should take serious care of the path of your critic and actor network.
lastest date:  2018.09.10
'''
################################################################################
import numpy as np
import torch as pt
#activation function
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import pickle
import torchvision as ptv
from memory import ReplayMemory, Experience

pt.cuda.set_device(1)
n_episode = 3000

max_steps = 100
n_agents = 4
batch_size = 600
pkl_file = open('data_saq.pkl', 'rb')

# should be unified when running in the server: which pkl file
memory = ReplayMemory(n_episode*n_agents*max_steps)


use_cuda = pt.cuda.is_available()


for i in range(n_episode):
    data1 = pickle.load(pkl_file)
    data2 = pickle.load(pkl_file)
    data3 = pickle.load(pkl_file)
    print ('episode is %d' % (i))
    for j in range(max_steps):
        #for k in range(n_agents):
        tmp_whole_obs = data1[j]
        tmp_whole_act = data2[j]
        memory.push(tmp_whole_obs, tmp_whole_act,'' , '','')


loss_func = pt.nn.MSELoss().cuda()

class meta_actor(pt.nn.Module):
    def __init__(self, dim_observation, dim_action):
        # print('model.dim_action',dim_action)
        super(meta_actor, self).__init__()
        self.FC1 = pt.nn.Linear(dim_observation, 500)
        self.FC2 = pt.nn.Linear(500, 128)
        self.FC3 = pt.nn.Linear(128, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result

model = meta_actor(22,2)
model.cuda()


optimizer = Adam(model.parameters(), lr=0.001)


for t in range(100000):

    ByteTensor = pt.cuda.ByteTensor if use_cuda else pt.ByteTensor
    FloatTensor = pt.cuda.FloatTensor if use_cuda else pt.FloatTensor


    transitions = memory.sample(batch_size)
    batch = Experience(*zip(*transitions))
    optimizer.zero_grad()

    state_batch = Variable(pt.stack(batch.states).type(FloatTensor))
    action_batch = Variable(pt.stack(batch.actions).type(FloatTensor))
    state_batch = state_batch.view(batch_size*4, -1)
    #whole_state = state_batch.view(batch_size, -1)
    whole_action = action_batch.view(batch_size*4, -1)/4

    sb = state_batch.detach()
    act = model(sb.unsqueeze(0)).squeeze()

    loss = loss_func(act, whole_action)

    print ('**********************************')
    print('Episode:%d,loss = %f' % (t, loss))
    loss.backward()
    optimizer.step()

    if t % 1000 == 0 and t > 0:
        pt.save(model, 'meta_actor/meta_actor[' + str(t) + '].pkl_episode' + str(t))





