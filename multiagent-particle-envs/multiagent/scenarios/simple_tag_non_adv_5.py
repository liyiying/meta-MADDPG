import numpy as np
from multiagent.core import World, Agent, Landmark, Target
from multiagent.scenario import BaseScenario

threshhold = 0.02


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 4+1
        num_target = 2
        num_adversaries = 0

        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.distance_rate = [float(0) for i in range(num_agents)]
        # add target of agents
        world.targets = [Target() for i in range(num_target)]
        for i, target in enumerate(world.targets):
            target.name = 'target %d' % i
            target.collide = False
            target.movable = False
            target.size = 0.07
            target.boundary = False
            target.entity_id = i + 1
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.agent_id = i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            # add the function of agent target
            if not agent.adversary:
                agent.target_id = i%2 + 1
                agent.collide_count = 0
                agent.distance_to_target = 0
                agent.if_reach = False
                agent.accu_dis = agent.size + world.targets[agent.target_id-1].size

            agent.size = 0.035 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.07
            landmark.boundary = False



        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        # No.5 agent as a newer
            if agent.agent_id >3:
                agent.color = np.array([0.65, 0.65, 0.65])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, target in enumerate(world.targets):
            target.color = np.array([0.25, 0.55, 0.78])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                #landmark.state.p_pos = np.array([0.9,0.9])
                landmark.state.p_vel = np.zeros(world.dim_p)

        int_u = np.random.randn(1)
        if int_u[0]>0:
            bool_test = False
        else:
            bool_test = True


        for i, target in enumerate(world.targets):
            if not target.boundary:
                target.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                '''
                if bool_test:
                    if i ==0:
                        target.state.p_pos = np.random.uniform(-0.7, -0.5, world.dim_p)
                    if i==1:
                        target.state.p_pos = np.random.uniform(+0.5, +0.7, world.dim_p)
                else:
                    if i ==1:
                        target.state.p_pos = np.random.uniform(-0.7, -0.5, world.dim_p)
                    if i==0:
                        target.state.p_pos = np.random.uniform(+0.5, +0.7, world.dim_p)
                '''

                #target.state.p_pos = np.array([0.9,0.9])

                #target.state.p_pos = np.array([0.9,0.9])
                target.state.p_vel = np.zeros(world.dim_p)

        for entity in world.targets:
            if not entity.boundary:
                for a in world.agents:
                    if a.target_id == entity.entity_id:
                        a.distance_to_target = np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos)))
                        a.accu_dis = a.size + world.targets[a.target_id-1].size
                        a.collide_count = 0


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        if dist < dist_min:
            agent1.collide_count +=1
            agent2.collide_count +=1

        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 1
        # we assume that good agents do not collide, but the penalty is relatively small
        if agent.collide:
            for agent1 in world.agents:
                if not agent1.adversary and agent.agent_id != agent1.agent_id:
                    if self.is_collision(agent, agent1):
                        rew -= 1


        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        agents = self.good_agents(world)
        # agents are rewarded for reaching the target
        tmp = 0
        tmpA = []

        train_or_test = world.train_or_test


        for entity in world.targets:
            if not entity.boundary:
                if agent.target_id == entity.entity_id:
                    tmp = np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos)))
                    tmpA.append(-0.2 * tmp)
                    dist_min = agent.size + entity.size
                    if tmp < dist_min + threshhold:
                        rew += 2
                        if train_or_test is True:
                            entity.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                            entity.state.p_vel = np.zeros(world.dim_p)
                        else:
                            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                            agent.state.p_vel = np.zeros(world.dim_p)
                            agent.state.c = np.zeros(world.dim_c)
                        world.distance_rate[agent.agent_id] = float(agent.distance_to_target / agent.accu_dis)
                        agent.if_reach = True
                        #for a in world.agents:
                        #     if a.target_id == entity.entity_id:
                        #         # initial the shortest distance between the agent and its target.
                        agent.distance_to_target = np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos)))
                        agent.accu_dis = agent.size + entity.size

        rew +=max(tmpA)
        # communication of all other agents

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        for entity in world.targets:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        tmp = 0
        max_i = 0



        for i, other in enumerate(world.agents):
            if other is agent: continue
            tmp_pos = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if tmp_pos > tmp:
                tmp = tmp_pos
                max_i = i

        agent.max_id = max_i


        for i, other in enumerate(world.agents):
            if other is agent or i == max_i: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            #if not other.adversary:
            other_vel.append(other.state.p_vel)
        '''
        tmp_state = []
        for i, other in enumerate(world.agents):
            if other is agent : continue
            tmp = []
            tmp_pos = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            tmp.append(i)
            tmp.append(tmp_pos)
            tmp_state.append(tmp)

        def takeSecond(elem):
            return elem[1]
        tmp_state.sort(key=takeSecond,reverse = False)

        for i,other in enumerate(world.agents):
            for j,tmppos in enumerate(tmp_state[3:len(world.agents)]):
                if other is agent or i == tmppos[0]: break

                if j == len(world.agents)-4-1:
                    comm.append(other.state.c)
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    # if not other.adversary:
                    other_vel.append(other.state.p_vel)
        '''
                    


        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
