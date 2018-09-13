from model import Critic, Actor
import torch as th
from copy import deepcopy
from torch.optim import Adam
from memory import ReplayMemory, Experience
from randomProcess import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import datetime

import torch.nn as nn
import numpy as np
from params import scale_reward


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train,initial_train,test_or_train):
        sum_obs_dim = 0
        self.actors = []
        self.critics = []
        # if we need to restart the training progress of network, we should inital the all agents networks.
        initial_train = initial_train
        # based on the design of meta_actor and meta_critic, here, we reset the initial function of initialation of Actor and Critic
        if initial_train is True:
            self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
            self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        else:
            self.actors = [Actor(dim_obs, dim_act) for i in range(4)]
            self.critics = [Critic(n_agents-1, dim_obs,
                               dim_act) for i in range(4)]

        if initial_train is True:
            self.actors_target = deepcopy(self.actors)
            self.critics_target = deepcopy(self.critics)
            self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
            self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act

        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.test = test_or_train
        if self.test is True:

            self.var = [0.0 for i in range(n_agents)]
        else:
            self.var = [0.5 for i in range(n_agents)]
            #self.var[4] = 1.0



        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            if initial_train is True:
                for x in self.actors_target:
                    x.cuda()
                for x in self.critics_target:
                    x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self,i_episode, initial_train):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []

        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = Variable(th.stack(batch.states).type(FloatTensor))

            action_batch = Variable(th.stack(batch.actions).type(FloatTensor))
            reward_batch = Variable(th.stack(batch.rewards).type(FloatTensor))

            s = [s for s in batch.next_states if s is not None]
            non_final_next_states = Variable(th.stack(s).type(FloatTensor))

            #tmp_whole_state = state_batch[:, (1, 4), :]
            initial_train = initial_train
            if agent == 4:
                tmp_state =  Variable(th.zeros(self.batch_size, 5,22).type(FloatTensor))
                tmp_action = Variable(th.zeros(self.batch_size, 5,2).type(FloatTensor))
                tmp_no_final_s = Variable(th.zeros(len(non_final_next_states), 5,22).type(FloatTensor))
                non_final_next_actions =Variable(th.zeros(len(non_final_next_states), 5,2).type(FloatTensor))
            else:
                tmp_state =  Variable(th.zeros(self.batch_size, 4,22).type(FloatTensor))
                tmp_action = Variable(th.zeros(self.batch_size, 4,2).type(FloatTensor))
                tmp_no_final_s = Variable(th.zeros(len(non_final_next_states), 4,22).type(FloatTensor))
                non_final_next_actions =Variable(th.zeros(len(non_final_next_states), 4,2).type(FloatTensor))

            non_final_next_actions_tmp = [  # [torch.FloatTensor of size 989x2]
                self.actors_target[i](non_final_next_states[:,  # [torch.FloatTensor of size 989x213]
                                      i,
                                      :]) for i in range(
                    self.n_agents)]
            non_final_next_actions_tmp = Variable(th.stack(non_final_next_actions_tmp).type(FloatTensor))
            non_final_next_actions_tmp = (
                non_final_next_actions_tmp.transpose(0,
                                                 1).contiguous())

            startTime = datetime.datetime.now()
            # the main difference between the double values of initial_train is: when initial_train is True, the input of the critic network
            # is all agents' observation, while it is False, the input is only the nearest four agents observation of the now agent
            if initial_train is False:
                #non_final_next_actions = []
                for j in range(self.batch_size):
                    if j < len(non_final_next_states):
                        tmp_no_final_s[j,0:4,:] = non_final_next_states[j, ([i for i in range(self.n_agents) if i != batch.max_id[j][agent]]), :]
                        #non_final_next_actions[j,:,:] = Variable(th.stack([self.actors_target[i](non_final_next_states[j, i, :]) for i in range(self.n_agents) if i!=batch.max_id[j][agent]]).type(FloatTensor))
                        non_final_next_actions[j,0:4,:] = non_final_next_actions_tmp[j, ([i for i in range(self.n_agents) if i != batch.max_id[j][agent]]), :]

                    tmp_state[j,0:4,:] = state_batch[j, ([i for i in range(self.n_agents) if i != batch.max_id[j][agent]]), :]
                    tmp_action[j,0:4,:] = action_batch[j, ([i for i in range(self.n_agents) if i != batch.max_id[j][agent]]), :]

            if agent ==4:
                tmp_state[:,4,:] = tmp_state[:,3,:]
                tmp_action[:,4,:] = tmp_action[:,3,:]
                tmp_no_final_s[:,4,:] = tmp_no_final_s[:,3,:]
                non_final_next_actions[:,4,:] = non_final_next_actions[:,3,:]


            if initial_train is True:
                tmp_state = state_batch
                tmp_action =action_batch
                tmp_no_final_s = non_final_next_states

            #whole_state = state_batch.view(self.batch_size, -1)
            #print('-----------------------------')
            whole_state = tmp_state.view(self.batch_size, -1)
            # print('whole_state',whole_state)  [torch.FloatTensor of size 100x62]
            whole_action = tmp_action.view(self.batch_size, -1)
            # non_final_next_states = non_final_next_states.view(next_state_count,-1)
            # print('non_final_next_states',non_final_next_states)
            self.critic_optimizer[agent].zero_grad()

            current_Q = self.critics[agent](whole_state, whole_action)


            if initial_train is True:
                non_final_next_actions = [  # [torch.FloatTensor of size 989x2]
                    self.actors_target[i](tmp_no_final_s[:,  # [torch.FloatTensor of size 989x213]
                                          i,
                                          :]) for i in range(
                        self.n_agents)]
                non_final_next_actions = Variable(th.stack(non_final_next_actions).type(FloatTensor))


            target_Q = Variable(th.zeros(
                self.batch_size, 1).type(FloatTensor))
            # print('non_final_mask',non_final_mask)

            if initial_train is True:
                target_Q[non_final_mask] = self.critics_target[agent](
                    tmp_no_final_s.view((-1, self.n_agents * self.n_states)),
                    non_final_next_actions.view((-1, self.n_agents * self.n_actions)))
            else:
                if agent != 4:
                    target_Q[non_final_mask] = self.critics_target[agent](
                            tmp_no_final_s.view((-1, (self.n_agents-1) * self.n_states)),
                            non_final_next_actions.view((-1, (self.n_agents-1) * self.n_actions)))
                else:
                    target_Q[non_final_mask] = self.critics_target[agent](
                        tmp_no_final_s.view((-1, self.n_agents  * self.n_states)),
                        non_final_next_actions.view((-1, self.n_agents  * self.n_actions)))


            # scale_reward: to scale reward in Q functions
            target_Q = (target_Q * self.GAMMA) + (
                reward_batch[:, agent].reshape( self.batch_size,1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]

            action_i = self.actors[agent](state_i)*4
            ac = tmp_action.clone()
            if initial_train is False and agent !=4:
                for j in range(self.batch_size):
                    if agent<batch.max_id[j][agent]:
                        tmp_agent = agent
                    else:
                        tmp_agent = agent-1
                    ac[j, tmp_agent, :] = action_i[j]
            if agent == 4:
                ac[:, 4, :] = action_i
                ac[:, 3, :] = action_i


            if initial_train is True:
                ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = Variable(th.zeros(
            self.n_agents,
            self.n_actions))
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            act += Variable(
                th.from_numpy(
                    np.random.randn(2) * self.var[i]).type(FloatTensor))

            if self.episode_done > self.episodes_before_train and \
                            self.var[i] > 0.005:
                self.var[i] *= 0.999998
            act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act          
        self.steps_done += 1

        return actions

    def select_action_test(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = Variable(th.zeros(
            self.n_agents,
            self.n_actions))
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()

            if self.episode_done > self.episodes_before_train and \
                            self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions