###############################################################################
'''
file name: evaluate.py
function: based on the running output of all agent actor network, we store the collide and accumulated distance of all agent;
            Then, we use the script evaluate.py to analyze different model results.
note: you should take serious care of the path of your critic and actor network.
lastest date:  2018.09.10
'''
################################################################################

import numpy as np
import pickle

activate_meta_actor = False
activate_idiot = False
activate_initial = True
count_episode = 1200

if activate_meta_actor is True:
    pkl_file = open('meta_figure/output/data_meta' + str(count_episode) + '.pkl', 'rb')
    txt_File = 'meta_figure/output/meta/data_meta' + str(count_episode) + '.txt'

if activate_idiot is True:
    pkl_file = open('meta_figure/output/data_idiot' + str(count_episode) + '.pkl', 'rb')
    txt_File = 'meta_figure/output/idiot/data_idiot' + str(count_episode) + '.txt'
if activate_initial is True:
    pkl_file = open('meta_figure/output/data_initial' + str(count_episode) + '.pkl', 'rb')
    txt_File = 'meta_figure/output/initial/data_initial' + str(count_episode) + '.txt'


data_0 = []
data_1 = []
data_2 = []
data_3 = []
data_4 = []
data_5 = []
data_6 = []
data_7 = []

num_agent = 5
count_agent = [0 for i in range(num_agent)]

data = pickle.load(pkl_file)
print('agent_id---i_episode----target_count ---- collide_count ----- distance_rate')
while data:
#for i in range(100):
    if data[4] >=1:
        data[4] = 0.99
    if data[0] ==0:
        data_0.append(data)
    elif data[0] ==1:
        data_1.append(data)
    elif data[0] ==2:
        data_2.append(data)
    elif data[0] ==3:
        data_3.append(data)
    elif data[0] ==4:
        data_4.append(data)
    elif data[0] ==5:
        data_5.append(data)
    elif data[0] ==6:
        data_6.append(data)
    elif data[0] ==7:
        data_7.append(data)

    count_agent[data[0]] +=1
    try:
        data = pickle.load(pkl_file)
    except EOFError:
        break

pkl_file.close()

average_collide = []

average_collide.append(np.sum(data_0,0)[3])
average_collide.append(np.sum(data_1,0)[3])
average_collide.append(np.sum(data_2,0)[3])
average_collide.append(np.sum(data_3,0)[3])
average_collide.append(np.sum(data_4,0)[3])
'''
if len(data_5):
    average_collide.append(0)
else:
    average_collide.append(np.sum(data_5,0)[3])
if len(data_6) == 0 :
    average_collide.append(0)
else:
    average_collide.append(np.sum(data_6, 0)[3])

average_collide.append(np.sum(data_7,0)[3])
'''

average_ratio_distance = []
#for i in range(num_agent):
#    average_ratio_distance.append((np.sum(data_0,0)[4]))
average_ratio_distance.append(np.sum(data_0,0)[4])
average_ratio_distance.append(np.sum(data_1,0)[4])
average_ratio_distance.append(np.sum(data_2,0)[4])
average_ratio_distance.append(np.sum(data_3,0)[4])
average_ratio_distance.append(np.sum(data_4,0)[4])
'''
if len(data_5):
    average_ratio_distance.append(0)
else:
    average_ratio_distance.append(np.sum(data_5,0)[4])

if len(data_6) == 0 :
    average_ratio_distance.append(0)
else:
    average_ratio_distance.append(np.sum(data_6,0)[4])
average_ratio_distance.append(np.sum(data_7,0)[4])
'''


f = file(txt_File, 'a+')

f.write('++++++++++++++++'+'\n')
f.write(str(count_agent)+'\n')
print(count_agent)
f.write('++++++++++++++++'+'\n')

print('average collision of agent')
newline = 'average collision of agent'+'\n'
f.write(newline)
print(average_collide)
f.write(str(average_collide)+'\n')
print('******************************')
f.write('******************************'+'\n')

for i in range(num_agent):
    tmp = average_collide[i]/count_agent[i]
    print('the agent id is %d, its average collision is %f',(i,tmp))
    newline = 'the agent id is '+str(i)+', its average collision is' + str(tmp)+'\n'
    f.write(newline)
print('average ratio between shortest distance and ture distance')
f.write('average ratio between shortest distance and ture distance'+'\n')
print(average_ratio_distance)
f.write(str(average_ratio_distance)+'\n')
print('******************************')
f.write('******************************'+'\n')
for i in range(num_agent):
    tmp = average_ratio_distance[i]/count_agent[i]
    print('the agent id is %d, its average ratio between shortest distance and ture distance is %f',(i,tmp))
    newline = 'the agent id is '+str(i)+', its average ratio between shortest distance and ture distance is' + str(tmp)+'\n'
    f.write(newline)

f.close()

