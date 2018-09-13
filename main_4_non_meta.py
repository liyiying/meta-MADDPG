###############################################################################
'''
file name: main_4_non_meta.py
function: This script is the main function of our method. In order to test the effect of meta actor and meta critic.
           Here we have three mode of running code.
            activate_meta_actor = True, means training all agents, and the added agent is meta mode.
            initial_train = False, means training all agent, and the added agent' critic network is idiot.
            test_initial = False, means training all agent from the initial stage.
note: you should take serious care of the path of your critic and actor network.
lastest date:  2018.09.10
'''
################################################################################

from torch.autograd import Variable
import numpy as np
import torch as th
import torch.nn.functional as F
from params import scale_reward
import datetime
import visdom
import imageio
import pickle
from copy import deepcopy
from torch.optim import Adam
import os, sys
print sys.path
import argparse
import multiagent
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from maddpg import MADDPG


#th.cuda.set_device(1)   # in order to run more instances, we will set the cuda devices Manually.

scenario = scenarios.load('/home/zw/lyy/maddpg/multiagent-particle-envs/multiagent/scenarios/simple_tag_non_adv_4.py').Scenario()

class meta_actor(th.nn.Module):
    def __init__(self, dim_observation, dim_action):
        # print('model.dim_action',dim_action)
        super(meta_actor, self).__init__()
        self.FC1 = th.nn.Linear(dim_observation, 500)
        self.FC2 = th.nn.Linear(500, 128)
        self.FC3 = th.nn.Linear(128, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result

model_actor = meta_actor(22,2)
model_actor.cuda()

class meta_critic(th.nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(meta_critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = self.dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = th.nn.Linear(obs_dim, 1024)
        self.FC2 = th.nn.Linear(1024 + act_dim, 512)
        self.FC3 = th.nn.Linear(512, 300)
        self.FC4 = th.nn.Linear(300, 1)

    # obs:batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))

model_critic = meta_critic(5, 22, 2)
model_critic.cuda()

# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                    shared_viewer=True)

n_agents = env.n
# n_states = env.observation_space
n_actions = world.dim_p
capacity = 1000000
batch_size = 100
totalTime = 0

vis = visdom.Visdom(port=8097)
win = None
param = None

np.random.seed(1234)
th.manual_seed(1234)

n_episode = 4000
max_steps = 100
episode_before_train = 100
obs = env.reset()
n_states = len(obs[0])

activate_meta_actor = True
initial_train = False
test_initial = False


test_or_train = False
world.train_or_test = not test_or_train
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episode_before_train,initial_train,test_or_train)



if test_initial is True:
    maddpg.actors.append(model_actor)
    maddpg.critics.append(model_critic)

    maddpg.actors_target = deepcopy(maddpg.actors)
    maddpg.critics_target = deepcopy(maddpg.critics)

    maddpg.critic_optimizer = [Adam(x.parameters(),
                                  lr=0.001) for x in maddpg.critics]
    maddpg.actor_optimizer = [Adam(x.parameters(),
                                 lr=0.0001) for x in maddpg.actors]


if activate_meta_actor is True and initial_train is False and test_initial is False:
    maddpg.actors.append(model_actor)
    maddpg.critics.append(model_critic)
    for i in range(maddpg.n_agents-1):
        maddpg.critics[i] = th.load('meta/critic_model/critic[' + str(i) + '].pkl_episode' + str(3000))
        maddpg.actors[i] = th.load('meta/actor_model/actors[' + str(i) + '].pkl_episode' + str(3000))

    maddpg.actors[n_agents-1] =th.load('meta/meta_actor.pkl_episode',map_location=lambda storage,loc:storage.cuda(0) )
    #maddpg.critics[n_agents-1] =th.load('meta/meta_critic.pkl_episode' + str(10000),map_location=lambda storage,loc:storage.cuda(0))
    maddpg.critics[n_agents-1] = th.load('meta/meta_critic.pkl_episode' ,map_location=lambda storage,loc:storage.cuda(0))

    maddpg.actors_target = deepcopy(maddpg.actors)
    maddpg.critics_target = deepcopy(maddpg.critics)

    maddpg.critic_optimizer = [Adam(x.parameters(),
                                  lr=0.001) for x in maddpg.critics]
    #maddpg.critic_optimizer[4] = Adam(maddpg.critics[4].parameters(), lr=0.0005)
    maddpg.actor_optimizer = [Adam(x.parameters(),
                                 lr=0.0001) for x in maddpg.actors]
    #maddpg.actor_optimizer[4] = Adam(maddpg.actors[4].parameters(),lr=0.00005)


if activate_meta_actor is False and initial_train is False and test_initial is False:
    maddpg.actors.append(model_actor)
    maddpg.critics.append(model_critic)
    for i in range(maddpg.n_agents-1):
        maddpg.critics[i] = th.load('meta/critic_model/critic[' + str(i) + '].pkl_episode' + str(3000))
        maddpg.actors[i] = th.load('meta/actor_model/actors[' + str(i) + '].pkl_episode' + str(3000))

    maddpg.actors[n_agents-1] =th.load('meta/meta_actor.pkl_episode',map_location=lambda storage,loc:storage.cuda(0) )
    #maddpg.critics[n_agents-1] =th.load('meta/meta_critic.pkl_episode' + str(10000),map_location=lambda storage,loc:storage.cuda(0))
    maddpg.critics[n_agents-1] = th.load('meta/meta_critic_0.pkl_episode' ,map_location=lambda storage,loc:storage.cuda(0))

    maddpg.actors_target = deepcopy(maddpg.actors)
    maddpg.critics_target = deepcopy(maddpg.critics)

    maddpg.critic_optimizer = [Adam(x.parameters(),
                                  lr=0.001) for x in maddpg.critics]
    maddpg.actor_optimizer = [Adam(x.parameters(),
                                 lr=0.00005) for x in maddpg.actors]
    maddpg.actor_optimizer[4] = Adam(x.parameters(),lr=0.00005)


FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

txt_File = 'new/output/test(copy)_meta_actor_reward'+str(initial_train)+str(activate_meta_actor)+str(test_initial)+'.txt'
f = file(txt_File, 'a+')

# Here, the additional agent model is agent 0 actor
output = open('new/output/data(copy)_meta_actor_test1'+str(initial_train)+str(activate_meta_actor)+str(test_initial)+'.pkl', 'wb')


f.write('the output of reward information of the training progress!'+'\n')
f.write('episode----reward----runtime'+'\n')

for i_episode in range(n_episode):
    startTime = datetime.datetime.now()
    obs = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    # obs = np.asarray(obs)
    reward_record = []
    adversaries_reward_record = []
    agent_reward_record = []
    frame_meta = []

    # for i in range(len(obs)):
    #     if isinstance(obs[i], np.ndarray):
    #         obs[i] = th.from_numpy(obs[i]).float()
    #         obs[i] = Variable(obs[i]).type(FloatTensor)

    total_reward = 0.0
    adversaries_reward = 0.0
    agent_reward = 0.0
    total_reward_5 = 0.0

    rr = np.zeros((n_agents,))
    # restore the frame of action
    frame = []
    for t in range(max_steps):
        # for j in range(len(obs)):
        #    obs = Variable(obs).type(FloatTensor)

        obs = Variable(obs).type(FloatTensor)
        agent_max_id = []
        for i in range(n_agents):
            agent_max_id.append(deepcopy(env.agents[i].max_id))

        action = maddpg.select_action(obs).data.cpu()
        #obs_, reward, done, _ = env.step(deepcopy(action.numpy()))
        obs_, reward, done, _ = env.step(action.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        # obs_ = np.asarray(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        #adversaries_reward += reward[0:5].sum()
        if initial_train is False:
            total_reward_5 +=reward[4]
        else:
            total_reward_5 += 0.0
        #agent_reward += reward[5:9].sum()
        rr += reward.cpu().numpy()


        maddpg.memory.push(obs.data, action, next_obs, reward, agent_max_id)

        obs = next_obs
        c_loss, a_loss = maddpg.update_policy(i_episode,initial_train)
        #frame.append(env.render())
        #env.render()
    #if i_episode == 1:
    #    a = np.array(frame)
    #    b = np.reshape(a, (600, 700, 700, 3))
    #    imageio.mimsave('test_adv.gif', b, 'GIF')
    if i_episode % 100 == 0 and i_episode > 0 and test_initial is False and initial_train is True:
        for i in range(maddpg.n_agents):
            th.save(maddpg.critics[i], 'new/model_new/critic[' + str(i) + '].pkl_episode' + str(i_episode))
            th.save(maddpg.actors[i], 'new/model_new/actors[' + str(i) + '].pkl_episode' + str(i_episode))
    if i_episode % 100 == 0 and i_episode > 0 and test_initial is True and initial_train is False:
        for i in range(maddpg.n_agents):
            th.save(maddpg.critics[i], 'new/model_initial/critic[' + str(i) + '].pkl_episode' + str(i_episode))
            th.save(maddpg.actors[i], 'new/model_initial/actors[' + str(i) + '].pkl_episode' + str(i_episode))
    if i_episode % 100 == 0 and i_episode > 0 and test_initial is False and activate_meta_actor is True and initial_train is False:
        for i in range(maddpg.n_agents):
            th.save(maddpg.critics[i], 'new/meta_model/critic[' + str(i) + '].pkl_episode' + str(i_episode))
            th.save(maddpg.actors[i], 'new/meta_model/actors[' + str(i) + '].pkl_episode' + str(i_episode))
    if i_episode % 100 == 0 and i_episode > 0 and test_initial is False and activate_meta_actor is False and initial_train is False:
        for i in range(maddpg.n_agents):
            th.save(maddpg.critics[i], 'new/meta_idiot/critic[' + str(i) + '].pkl_episode' + str(i_episode))
            th.save(maddpg.actors[i], 'new/meta_idiot/actors[' + str(i) + '].pkl_episode' + str(i_episode))
    maddpg.episode_done += 1
    endTime = datetime.datetime.now()
    runTime = (endTime - startTime).seconds
    totalTime = totalTime + runTime

    frame_meta.append(i_episode)
    total_reward_cpu = float(total_reward.data.cpu())
    if initial_train is False:
        total_reward_5_cpu = float(total_reward_5.data.cpu())
    else:
        total_reward_5_cpu = 0.0
    frame_meta.append(total_reward_cpu)
    frame_meta.append(total_reward_5_cpu)
    frame_meta.append(runTime)
    pickle.dump(frame_meta, output)

    print('Episode:%d,reward = %f' % (i_episode, total_reward))
    f.write('Episode:'+str(i_episode)+',reward = '+str(total_reward_cpu)+'\n')
    #print('Episode:%d,agent  = %f' % (i_episode, adversaries_reward))
    print('Episode:%d,agent_5_reward = %f' % (i_episode,total_reward_5_cpu ))
    print('this episode run time:' + str(runTime))
    print('totalTime:' + str(totalTime))
    reward_record.append(total_reward)
    adversaries_reward_record.append(adversaries_reward)
    agent_reward_record.append(agent_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on WaterWorld\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % n_agents)


f.close()
output.close()