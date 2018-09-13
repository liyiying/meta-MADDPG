###############################################################################
'''
file name: meta_critic_rnn.py
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
import datetime

pt.cuda.set_device(1)
STATE_DIM = 22
TASK_CONFIG_DIM = 3
ACTION_DIM = 2

n_episode = 2       #1000
max_steps = 100
n_agents = 4
length_lstm = 10
pkl_file = open('data_saq.pkl', 'rb')


# should be unified when running in the server: which pkl file
memory = ReplayMemory(n_episode*n_agents*max_steps+100)

use_cuda = pt.cuda.is_available()
ByteTensor = pt.cuda.ByteTensor if use_cuda else pt.ByteTensor
FloatTensor = pt.cuda.FloatTensor if use_cuda else pt.FloatTensor

for i in range(n_episode):
    data1 = pickle.load(pkl_file)
    data2 = pickle.load(pkl_file)
    data3 = pickle.load(pkl_file)
    print ('episode is %d' % (i))
    for j in range(max_steps):
        memory.push(data1[j], data2[j], '', data3[j],'')


loss_func = pt.nn.MSELoss().cuda()

class meta_critic(pt.nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action, configs):
        super(meta_critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = self.dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = pt.nn.Linear(obs_dim, 1024)
        self.FC2 = pt.nn.Linear(1024 + act_dim + configs, 512)
        self.FC3 = pt.nn.Linear(512, 300)
        self.FC4 = pt.nn.Linear(300, 1)

    # obs:batch_size * obs_dim
    def forward(self, obs, acts, config):
        result = F.relu(self.FC1(obs))
        combined = pt.cat([result, acts, config], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Meta_Critic_ConfigNetwork(pt.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Meta_Critic_ConfigNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = pt.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = pt.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial states
        h0 = Variable(pt.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(pt.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

model = meta_critic(n_agents, STATE_DIM, ACTION_DIM, TASK_CONFIG_DIM)
model.cuda()

task_config_input_dim = STATE_DIM + ACTION_DIM
config_network = Meta_Critic_ConfigNetwork(input_size = task_config_input_dim,hidden_size = 30,num_layers = 1,output_size = TASK_CONFIG_DIM)
config_network.cuda()

optimizer_meta_critic = Adam(model.parameters(), lr=0.001)
optimizer_config_network = Adam(config_network.parameters(), lr=0.001)



for t in range(100000):

    ByteTensor = pt.cuda.ByteTensor if use_cuda else pt.ByteTensor
    FloatTensor = pt.cuda.FloatTensor if use_cuda else pt.FloatTensor

    random_position = np.random.randint(low=length_lstm,high=min(memory.__len__(),n_episode*n_agents*max_steps))
    memory_info = memory.get_item(random_position, length_lstm)
    batch = Experience(*zip(*memory_info))
    state_batch = Variable(pt.stack(batch.states).type(FloatTensor))
    action_batch = Variable(pt.stack(batch.actions).type(FloatTensor))
    Q_batch = Variable(pt.stack(batch.rewards).type(FloatTensor))

    for i in range(n_agents):
        optimizer_meta_critic.zero_grad()

        whole_state = state_batch[0:length_lstm-1,i,:].view(length_lstm-1, 22)
        whole_action = action_batch[0:length_lstm-1,i,:].view(length_lstm-1, 2)/4
        final_state = state_batch[length_lstm-1,:,:]
        final_action = action_batch[length_lstm-1,:,:]

        pre_data_samples = pt.cat((whole_state, whole_action),1).unsqueeze(0)

        config = config_network(pre_data_samples)

        prediction = model(final_state.view(1,-1), final_action.view(1,-1), config.detach())
        loss_Q = loss_func(prediction, Q_batch[length_lstm-1,i].view(1,-1))
        loss_Q.backward()
        optimizer_meta_critic.step()

    print ('**********************************')
    print('Episode:%d,loss = %f' % (t, loss_Q))


    if (t+1) % 10000 == 0 and t > 0:
        pt.save(model, 'meta_critic/meta_critic[' + str(t+1) + '].pkl_episode' + str(t+1))







