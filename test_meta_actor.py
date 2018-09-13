###############################################################################
'''
file name: test_meta_actor.py
function: Based on the trained multiple agent models, multi-agent actors implement in a specific environment,
           and  we will statistic on the number of collisions and shortest distance ratio in different modes.
note: you should take serious care of the path of your critic and actor network.
lastest date:  2018.09.10
'''
################################################################################

import argparse

#import multiagent
import os, sys

sys.path.append('/home/lyy/Desktop/maddpg/multiagent-particle-envs')

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from maddpg import MADDPG

import numpy as np
import torch as pt
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
import datetime
import pickle
import imageio
pt.cuda.set_device(1)

scenario = scenarios.load('/home/zw/lyy/maddpg/multiagent-particle-envs/multiagent/scenarios/simple_tag_non_adv_4.py').Scenario()


# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                    shared_viewer=True)
n_agents = env.n
# n_states = env.observation_space
n_actions = world.dim_p
capacity = 1000000
batch_size = 1000
totalTime = 0

#vis = visdom.Visdom(port=8097)
win = None
param = None

np.random.seed(1234)
pt.manual_seed(1234)

n_episode = 100
max_steps = 600
episode_before_train = 1
obs = env.reset()
n_states = len(obs[0])

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

model_actor = meta_actor(22,2)
model_actor.cuda()

initial_train = False
test_or_train = True
world.train_or_test = not test_or_train
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episode_before_train,initial_train,test_or_train)


activate_meta_actor = True
activate_idiot = False
activate_initial = False
count_episode = 200

if activate_idiot is True or activate_meta_actor is True:
    maddpg.actors.append(model_actor)

if activate_meta_actor is True:
    #output = open('meta_figure/output/data_meta'+str(count_episode)+'.pkl', 'wb')
    output = open('test.pkl', 'wb')
    for i in range(maddpg.n_agents-1):
        maddpg.actors[i] = pt.load('meta_figure/meta/actor_'+str(count_episode)+'/actors[' + str(i) + '].pkl_episode' + str(count_episode),map_location=lambda storage,loc:storage.cuda(1))

if activate_idiot is True:
    output = open('meta_figure/output/data_idiot' + str(count_episode) + '.pkl', 'wb')
    for i in range(maddpg.n_agents):
        maddpg.actors[i] = pt.load('meta_figure/meta_and_idiot/actor_'+str(count_episode)+'/actors[' + str(i) + '].pkl_episode' + str(count_episode),map_location=lambda storage,loc:storage.cuda(1))

if activate_initial is True:
    output = open('meta_figure/output/data_initial' + str(count_episode) + '.pkl', 'wb')
    for i in range(maddpg.n_agents):
        maddpg.actors[i] = pt.load('meta_figure/initial_5/actor_'+str(count_episode)+'/actors[' + str(i) + '].pkl_episode' + str(count_episode),map_location=lambda storage,loc:storage.cuda(1))

FloatTensor = pt.cuda.FloatTensor if maddpg.use_cuda else pt.FloatTensor


for i_episode in range(n_episode):
    startTime = datetime.datetime.now()
    obs = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = pt.from_numpy(obs).float()
    # obs = np.asarray(obs)

    frame_gif = []
    # restore the information of agent id, ratio of distance , and collise_count
    tmp_id = 0
    tmp_ratio = 0
    tmp_colli = 0

    end_count_agent = [0 for i in range(n_agents)]


    for t in range(max_steps):
        obs = Variable(obs).type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = env.step(action.numpy())

        reward = pt.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        # obs_ = np.asarray(obs_)
        obs_ = pt.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        obs = next_obs
        if i_episode == 0:
            if t>1:
                frame_gif.append(env.render())
            if t == 101:
                a = np.array(frame_gif)
                b = np.reshape(a, (100, 700, 700, 3))
                if activate_meta_actor == True:
                    imageio.mimsave('meta_figure/gif/test_meta_actor' + str(count_episode) + '.gif', b, 'GIF')
                if activate_idiot == True:
                    imageio.mimsave('meta_figure/gif/test_idiot' + str(count_episode) + '.gif', b, 'GIF')
                if activate_initial == True:
                    imageio.mimsave('meta_figure/gif/test_initial' + str(count_episode) + '.gif', b, 'GIF')

        print('***********************************************************')
        for i, agent in enumerate(world.agents):
            if agent.if_reach is True:
                tmp_id = agent.agent_id
                tmp_colli = agent.collide_count
                tmp_ratio = world.distance_rate[agent.agent_id]
                frame = []
                end_count_agent[i] +=1
                frame.append(tmp_id)
                frame.append(i_episode)
                frame.append(end_count_agent[i])
                frame.append(tmp_colli)
                frame.append(tmp_ratio)
                print('The episode :%d ,The ratio between the travel distance and the shortest distance of agnet %d is: %f' % (i_episode,i, world.distance_rate[agent.agent_id]))
                print('The collision number of agent %d is: %f' % (i, agent.collide_count))
                agent.if_reach = False
                agent.collide_count = 0
                world.distance_rate[agent.agent_id] = 0
                pickle.dump(frame, output)
        print('***********************************************************')

    maddpg.episode_done += 1
    endTime = datetime.datetime.now()
    runTime = (endTime - startTime).seconds
    totalTime = totalTime + runTime
    print('Episode:%d,' % i_episode )
    print('this episode run time:' + str(runTime))
    print('totalTime:' + str(totalTime))

output.close()