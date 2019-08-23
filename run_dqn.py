"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
PyTorch: 0.4.0
gym: 0.8.0
"""


import gym
from dqn_per import DQN_PER
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time



env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 10000


time_start = time.time()

# RL_natural = DQN_PER(
#         n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.00005, prioritized=False,
#     )
#
# RL_natural_2 = DQN_PER(
#         n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.00005, prioritized=False, doubled_q= True
#     )

RL_prio = DQN_PER(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, prioritized=True
    )

# RL_prio_2 = DQN_PER(
#         n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.00005, prioritized=True, doubled_q= True
#     )
#
# RL_prio_selu = DQN_PER(
#         n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.00005, prioritized=True, relu_flag= False
#     )

# time_start = time.time()

def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in tqdm(range(10)):
        observation = env.reset()
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done:
                reward = 10

            RL.store_transition(observation, action, reward, observation_)


            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1

            if total_steps % 1000 == 0:
                print("total_steps:%d" %total_steps)

            if total_steps == 30000:
                time_end = time.time()
                print('total_time:', time_end-time_start)

    return np.vstack((episodes, steps))


# his_natural = train(RL_natural)
# his_natural_2 = train(RL_natural_2)
his_prio = train(RL_prio)
# his_prio_2 = train(RL_prio_2)
# his_prio_selu = train(RL_prio_selu)

# compare based on first success
# plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN(relu)')
# plt.plot(his_natural_2[0, :], his_natural_2[1, :] - his_natural_2[1, 0], c='y', label='Double DQN(relu)')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay(relu)') # 每个数据都以第一个数据为基准,看看差值是多少
# plt.plot(his_prio_2[0, :], his_prio_2[1, :] - his_prio_2[1, 0], c='g', label='Double DQN with prioritized replay(relu)')
# plt.plot(his_prio_selu[0, :], his_prio_selu[1, :] - his_prio_selu[1, 0], c='purple', label='DQN with prioritized replay(selu)')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()


