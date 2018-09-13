###############################################################################
'''
file name: meta_critic.py
function: Train a generic agent meta critic model with meta properties based on the four agents that have been trained on the premise.
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
n_episode = 1000
max_steps = 100
n_agents = 4
dim_observation = 22
dim_action = 2
batch_size = 600
pkl_file = open('data_saq.pkl', 'rb')
memory = ReplayMemory(n_episode*n_agents*max_steps)


use_cuda = pt.cuda.is_available()
ByteTensor = pt.cuda.ByteTensor if use_cuda else pt.ByteTensor
FloatTensor = pt.cuda.FloatTensor if use_cuda else pt.FloatTensor

for i in range(n_episode):
    data1 = pickle.load(pkl_file)
    data2 = pickle.load(pkl_file)
    data3 = pickle.load(pkl_file)
    print ('episode is %d' % (i))
    for j in range(max_steps):
        for k in range(n_agents):
            tmp_state = Variable(pt.zeros( 5, 22).type(FloatTensor))
            tmp_action = Variable(pt.zeros(5, 2).type(FloatTensor))
            tmp_state[0:4,:] = data1[j]
            tmp_state[4,:] = data1[j][k,:]
            tmp_action[0:4, :] = data2[j]
            tmp_action[4, :] = data2[j][k, :]

            memory.push(tmp_state, tmp_action, '', data3[j][k].cpu(),'')


loss_func = pt.nn.MSELoss().cuda()

class meta_critic(pt.nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(meta_critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = self.dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = pt.nn.Linear(obs_dim, 1024)
        self.FC2 = pt.nn.Linear(1024 + act_dim, 512)
        self.FC3 = pt.nn.Linear(512, 300)
        self.FC4 = pt.nn.Linear(300, 1)

    # obs:batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = pt.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))

model = meta_critic(5, 22, 2)
model.cuda()


optimizer = Adam(model.parameters(), lr=0.0001)



for t in range(100000):

    transitions = memory.sample(batch_size)
    batch = Experience(*zip(*transitions))
    optimizer.zero_grad()

    state_batch = Variable(pt.stack(batch.states).type(FloatTensor))
    action_batch = Variable(pt.stack(batch.actions).type(FloatTensor))

    Q_batch = Variable(pt.stack(batch.rewards).type(FloatTensor))

    whole_state = state_batch.view(batch_size, -1)
    whole_action = action_batch.view(batch_size, -1)
    whole_Q = Q_batch.view(batch_size, -1)

    prediction = model(whole_state, whole_action)

    #target_Q = Variable(pt.zeros(batch_size, 1).type(FloatTensor))

    loss_Q = loss_func(prediction, whole_Q)

    print ('**********************************')
    print('Episode:%d,loss = %f' % (t, loss_Q))
    loss_Q.backward()
    optimizer.step()

    if (t+1) % 1000 == 0 and t > 0:
        pt.save(model, 'meta_critic/meta_critic[' + str(t+1) + '].pkl_episode' + str(t+1))







