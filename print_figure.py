###############################################################################
'''
file name: print_figure.py
function: According to the sampling results of the three modes during the training process, the main function here is to draw their reward curves.
note: you should take serious care of the path of your critic and actor network.
lastest date:  2018.09.10
'''
################################################################################
import pickle
import numpy as np
import matplotlib.pyplot as plt

pkl_file1 = open('meta_figure/meta/data_meta_actor_test1FalseTrueFalse.pkl', 'rb')
pkl_file2 = open('meta_figure/initial_5/data_meta_actor_test1TrueFalseFalse.pkl', 'rb')
pkl_file3 = open('meta_figure/meta_and_idiot/data_meta_actor_test1FalseFalseFalse.pkl', 'rb')


data1 = pickle.load(pkl_file1)
data2 = pickle.load(pkl_file2)
data3 = pickle.load(pkl_file3)

data_meta = []
data_idiot = []
data_initial = []

count_agent = [0 for i in range(3)]

#while data1:
for i in range(1200):
    data_meta.append(data1)
    data_initial.append(data2)
    data_idiot.append(data3)
    for i in range(3):
        count_agent[i] +=1
    try:
        data1 = pickle.load(pkl_file1)
        data2 = pickle.load(pkl_file2)
        data3 = pickle.load(pkl_file3)

    except EOFError:
        break


test1 = np.array(data_meta[0:700])
test2 = np.array(data_initial[0:700])
test3 = np.array(data_idiot[0:700])

tmp1 = []
tmp2 = []
tmp3 = []
count_num = []

all_reward = False
ratio = 1
num_step = int(700/ratio)

for i in range(num_step):
    if all_reward is True:
        tmp1.append(test1[i*ratio][1])
        tmp2.append(test2[i * ratio][1])
        tmp3.append(test3[i * ratio][1])
    else:
        tmp1.append(test1[i*ratio][2])
        tmp2.append(test2[i * ratio][2])
        tmp3.append(test3[i * ratio][2])
    count_num.append(i*ratio)

tmp1 = np.array(tmp1)
tmp2 = np.array(tmp2)
tmp3 = np.array(tmp3)
count_num = np.array(count_num)


print('------------------------')

fig = plt.figure()

tmp_set = False

fig = plt.figure()
plt.plot(count_num, tmp1,'r',linewidth=1)
if tmp_set is True:
    plt.plot(count_num, tmp2,'g',linewidth=1)
plt.plot(count_num, tmp3,'b',linewidth=1)
plt.xlabel('Episode', fontsize=16)
if all_reward is False:
    plt.ylabel('Reward of the new robot', fontsize=16)
else:
    plt.ylabel('Reward', fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


if tmp_set is True:
    plt.legend(['Meta-Critic', 'Initial-Train', 'Idiot-Critic'],prop={'size':16} )
else:
    plt.legend(['Meta-Critic', 'Idiot-Critic'],loc='lower right',prop={'size':16})



if all_reward is False and tmp_set is True:
    plt.savefig('meta_figure/figure/5_reward.png', dpi=400)
if all_reward is True and tmp_set is True:
    plt.savefig('meta_figure/figure/All_reward.png', dpi=400)

if tmp_set is False:
    plt.savefig('meta_figure/figure/meta_and_idiot_reward.png', dpi=300)


plt.show()


pkl_file1.close()
pkl_file2.close()