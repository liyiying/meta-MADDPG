###############################################################################
'''
file name: collect_saq_non_adv.py
function: the [state, action, Q] resource of the next stage meta training.
note: you should take serious care of the path of your critic and actor network.
lastest date:  2018.09.10
'''
################################################################################

import datetime

import multiagent.scenarios as scenarios
import numpy as np
import torch as th
import visdom
from multiagent.environment import MultiAgentEnv
from torch.autograd import Variable
import pickle

from maddpg import MADDPG
# load the setting of the environment.
scenario = scenarios.load('/home/zw/lyy/maddpg/multiagent-particle-envs/multiagent/scenarios/simple_tag_non_adv_4.py').Scenario()

output = open('data_saq_test.pkl', 'wb')

# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                    shared_viewer=True)
#
world.train_or_test = True
n_agents = env.n
# some initial training parameters
n_actions = world.dim_p
# the capacity of the experience memory
capacity = 1000000


batch_size = 1000
totalTime = 0
n_episode = 3000
max_steps = 100
# before training, we will store the experience of all agents' state information for the next training process.
episode_before_train = 100
obs = env.reset()
n_states = len(obs[0])
initial_train = True
test_or_train = True

#vis = visdom.Visdom(port=8097)
win = None
param = None

np.random.seed(1234)
th.manual_seed(1234)
# the initial of the original maddpg, it is the basic part of our architecture.
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episode_before_train,initial_train,test_or_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

for i in range(maddpg.n_agents):
    maddpg.critics[i] = th.load('new/model_initial/critic[' + str(i) + '].pkl_episode' + str(3000))
    maddpg.actors[i] = th.load('new/model_initial/actors[' + str(i) + '].pkl_episode' + str(3000))

for i_episode in range(n_episode):
    startTime = datetime.datetime.now()
    obs = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()

    state_frame = []
    action_frame = []
    Q_frame = []

    # restore the frame of action
    frame = []
    for t in range(max_steps):
        obs = Variable(obs).type(FloatTensor)
        action = maddpg.select_action_test(obs).data.cpu()
        obs_, reward, done, _ = env.step(action.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = obs_

        q_value = []
        action = th.FloatTensor(action).type(FloatTensor)
        state_frame.append(obs)
        action_frame.append(action)
        for i in range(maddpg.n_agents):
            # we dump the [state,action] information of each agent into the pickle file.
            whole_state = obs.view(1, -1)
            whole_action = action.view(1, -1)
            q_tmp = maddpg.critics[i](whole_state, whole_action)
            q_value.append(q_tmp)

        q_value = th.FloatTensor(q_value).type(FloatTensor)
        Q_frame.append(q_value)
        obs = next_obs

    pickle.dump(state_frame, output)
    pickle.dump(action_frame, output)
    pickle.dump(Q_frame, output)

    del state_frame, action_frame, Q_frame

    maddpg.episode_done += 1
    endTime = datetime.datetime.now()
    runTime = (endTime - startTime).seconds
    totalTime = totalTime + runTime
    print('this episode is:' + str(i_episode))
    print('totalTime:' + str(totalTime))

output.close()