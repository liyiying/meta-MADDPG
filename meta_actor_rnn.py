###############################################################################
'''
file name: meta_actor_rnn.py
function: Train a generic agent metat actor model with meta properties based on the four agents that have been trained on the premise.
           Here, this code has contained the rnn structure.
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
STATE_DIM = 22
TASK_CONFIG_DIM = 3
ACTION_DIM = 2

n_episode = 2   #1000
max_steps = 100
n_agents = 4
length_lstm = 10
pkl_file = open('data_saq.pkl', 'rb')


# should be unified when running in the server: which pkl file
memory = ReplayMemory(n_episode*n_agents*max_steps+100)

use_cuda = pt.cuda.is_available()

for i in range(n_episode):
    data1 = pickle.load(pkl_file)
    data2 = pickle.load(pkl_file)
    data3 = pickle.load(pkl_file)
    print ('episode is %d' % (i))
    for j in range(max_steps):
        memory.push(data1[j], data2[j],'' , '','')


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



class Meta_Acotr_ConfigNetwork(pt.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Meta_Acotr_ConfigNetwork, self).__init__()
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


meta_actor_input_dim =  STATE_DIM + TASK_CONFIG_DIM
task_config_input_dim = STATE_DIM


config_network = Meta_Acotr_ConfigNetwork(input_size = task_config_input_dim,hidden_size = 30,num_layers = 1,output_size = TASK_CONFIG_DIM)
model = meta_actor(meta_actor_input_dim,ACTION_DIM)
config_network.cuda()
model.cuda()

optimizer_meta_actor = Adam(model.parameters(), lr=0.001)
optimizer_config_network = Adam(config_network.parameters(), lr=0.001)


for t in range(100000):

    ByteTensor = pt.cuda.ByteTensor if use_cuda else pt.ByteTensor
    FloatTensor = pt.cuda.FloatTensor if use_cuda else pt.FloatTensor

    random_position = np.random.randint(low=length_lstm,high=min(memory.__len__(),n_episode*n_agents*max_steps))
    memory_info = memory.get_item(random_position, length_lstm)
    batch = Experience(*zip(*memory_info))
    state_batch = Variable(pt.stack(batch.states).type(FloatTensor))
    action_batch = Variable(pt.stack(batch.actions).type(FloatTensor))


    for i in range(n_agents):

        optimizer_meta_actor.zero_grad()
        whole_state = state_batch[0:length_lstm-1,i,:].view(length_lstm-1, 22)
        whole_action = action_batch[0:length_lstm-1,i,:].view(length_lstm-1, 2)/4
        final_state = state_batch[length_lstm-1,i,:]
        final_action = action_batch[length_lstm-1,i,:]

        #pre_data_samples = pt.cat((whole_state, whole_action),1).unsqueeze(0)
        pre_data_samples = whole_state.unsqueeze(0)
        config = config_network(Variable(pre_data_samples).cuda()).squeeze()

        value_inputs = pt.cat((final_state, config.detach()), 0)
        act = model(value_inputs)*4

        loss = loss_func(act, final_action)
        loss.backward()
        optimizer_meta_actor.step()

    print ('**********************************')
    print('Episode:%d,loss = %f' % (t, loss))


    if t % 10000 == 0 and t > 0:
        pt.save(model, 'meta_actor/meta_actor[' + str(t) + '].pkl_episode' + str(t))






