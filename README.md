# meta-MADDPG

## Introduction

This is the code for implementing the meta-MADDPG algorithm presented in the paper: Improving Scalability in Applying
Reinforcement Learning into Multi-robot Scenarios. It is configured to be run in conjunction with environments from
the Multi-Agent Particle Environments (MPE).

Paper : [Improving Scalability in Applying Reinforcement Learning into Multi-robot Scenarios].

Environment : [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs).
(Training and testing is based on an instance of the environment named "simple_tag_non_adv_4.py").


## Dependency

- [pytorch](https://github.com/pytorch/pytorch)
- [visdom](https://github.com/facebookresearch/visdom)
- python 2

## Install

1. Build MPE environment.
  ```Shell
  # goto the path of multiagent-particle-envs
  cd multiagent-particle-envs
  # build MPE
  python setup.py install
  # (optional) if you change the code under the path of MPE, you can rebuild it, or delete it
  rm -rf build
  pip uninstall multiagent
  python setup.py install
  ```

2. Execute the main program and train a model of 4 agents or 5 agents

   **Note 1:** :You need to pay special attention to the file paths in your code and adjust the different execution modes as needed.

    ```Shell
    # Moreover, you can change the running mode through changing the code of
    #    activate_meta_actor = True
    #    initial_train       = False
    #    test_initial        = False
    python main_4_non_meta.py
    ```

3. Training the model of meta actor and meta critic

    **Note 1:** :You need to pay special attention to the file paths in your code and adjust the different execution modes as needed.

    **Note 2:** :According to the design needs, our code contains two modes of meta, one of which has a rnn structure, anyway, no.
    ```Shell
    python meta_actor.py   #  or python meta_actor_rnn.py
    python meta_critic.py  #  or python meta_critic_rnn.py
    make
    ```

4. Evaluate the meta model and make a figure
    ```Shell
    # On the premise of the completion of the training, we cancel the random action process,
    # run the actor model of each agent, and obtain the specific execution result.
    python test_meta_actor.py
    # Evaluate the mode of each mode, the statistical results mainly include the number of collisions and the shortest distance ratio:
    python evaluate.py
    # we can output the figure of results finally.
    python print_figure.py


## Result

five green spots are agents, black spots are obstacles, blue spots are targets, and gray for newcomer.
Meta-application: when newcomers are into the environment, the meta-actor network in the cloud can be downloaded to the newcomers to take
emergent and suitable actions directly.
- Four trained agents implementations:
![image](https://github.com/liyiying/meta-MADDPG/blob/master/meta_figure/gif/test_only_agent.gif)：
- idiot newcomer (The fifth agent arrives, and its actor network is idiot):
![image](https://github.com/liyiying/meta-MADDPG/blob/master/meta_figure/gif/idiot_agent_5.gif)
- meta newcomer (The fifth agent arrives, and its acotr network directly loads meta actor network)
![image](https://github.com/liyiying/meta-MADDPG/blob/master/meta_figure/gif/meta/test_meta1200.gif)：





