import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay import Memory
import time

class Net(nn.Module):
    def __init__(self,
                 n_actions,
                 n_features,
                 hidden_size = 20,          # 隐层节点数量
                 relu_flag = True,
                 ):

        # 父类初始化
        nn.Module.__init__(self)

        # 参数初始化
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.relu_flag = relu_flag

        # 构造神经网络
        self.fc1 = nn.Linear(self.n_features, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.n_actions)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc1.bias.data.fill_(0.1)
        self.out.weight.data.normal_(0, 0.3)
        self.out.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        if self.relu_flag:
            x = F.relu(x)
        else:
            x = F.selu(x)
        x = self.out(x)

        return x

class DQN_PER(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.005,       # 学习率
                 reward_decay=0.9,          # 折扣率
                 e_greedy=0.9,              # epison-soft策略中的epsilon
                 replace_target_iter=500,   # 更新网络的操作
                 memory_size=10000,         # 记忆库大小
                 batch_size=32,             # 批数据大小
                 hidden_size=20,            # 隐层节点数量
                 e_greedy_increment=None,
                 prioritized=True,          # 默认为优先经验回放
                 doubled_q = False,         # 默认不是用Double DQN
                 relu_flag=True
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy     # 默认为0.9
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.prioritized = prioritized    # 是否使用优先级回放
        self.hidden_size = hidden_size
        self.doubled_q = doubled_q
        self.relu_flag = relu_flag

        self.learn_step_counter = 0

        self.eval_net = Net(self.n_actions, self.n_features, relu_flag = self.relu_flag)
        self.target_net = Net(self.n_actions, self.n_features, relu_flag = self.relu_flag)

        # 根据实际情况建立记忆库
        if self.prioritized:
            self.memory = Memory(capacity=self.memory_size)
        else:
            self.memory = np.zeros((self.memory_size, self.n_features*2+2))

        # 定义优化器
        self.optimizer = t.optim.RMSprop(self.eval_net.parameters(), lr=self.lr)

        # 损失函数在后面定义
        self.loss_func = nn.MSELoss()
        self.cost_his = []

    # 存储经验
    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    # 行动策略 epsilon-soft policy
    def choose_action(self, observation):
        # time_start = time.time()
        x = t.unsqueeze(t.FloatTensor(observation), 0)
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.forward(x)
            action = np.argmax(actions_value.detach().numpy())
            # action = action[0]
            # time_end = time.time()
            # print('transfer time', time_end - time_start)
            # print('haha')
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 替换目标网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')

        # 经验采样
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
            ISWeights = t.FloatTensor(ISWeights)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
        b_s = t.FloatTensor(batch_memory[:, :self.n_features])
        b_s_ = t.FloatTensor(batch_memory[:, -self.n_features:])
        b_a = t.LongTensor(batch_memory[:, self.n_features].astype(int)).unsqueeze(dim=1)
        b_r = t.FloatTensor(batch_memory[:, self.n_features + 1]).unsqueeze(dim=1)

        # 计算Q(s,a), Q(s_,a_)
        q_eval = self.eval_net(b_s).gather(dim = 1, index = b_a)
        q_eval_next = self.eval_net(b_s_)
        max_next_action = q_eval_next.max(dim=1)[1].unsqueeze(dim=1)
        q_target_next = self.target_net(b_s_).detach()


        # 计算q_target
        if self.doubled_q:
            q_target = b_r + self.gamma * q_target_next.gather(dim = 1, index = max_next_action).view(self.batch_size, 1)
        else:
            q_target = b_r + self.gamma * q_target_next.max(1)[0].view(self.batch_size, 1)

        # 训练神经网络
        if self.prioritized:

            # 计算绝对误差(td-error)
            abs_errors = t.abs(q_target - q_eval).detach()

            loss = t.mean(ISWeights * t.pow(q_target - q_eval, 2))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新优先级
            self.memory.batch_update(tree_idx, abs_errors.numpy())     # update priority
        else:
            loss = t.mean(t.pow(q_target - q_eval, 2))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 添加损失函数信息
        self.cost_his.append(loss)

        # 更新epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # 增加学习步长
        self.learn_step_counter += 1