import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorNet(nn.Module):
    def __init__(self,
                 n_actions,
                 n_features,
                 hidden_size=40,  # 隐层节点数量
                 ):
        # 父类初始化
        nn.Module.__init__(self)

        # 参数初始化
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_size = hidden_size

        # 构造神经网络
        self.fc1 = nn.Linear(self.n_features, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.n_actions)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=1)

        return out

class CriticNet(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_size=20,  # 隐层节点数量
                 ):
        # 父类初始化
        nn.Module.__init__(self)

        # 参数初始化
        self.n_features = n_features
        self.hidden_size = hidden_size

        # 构造神经网络
        self.fc1 = nn.Linear(self.n_features, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

class A2C_Seperated(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 lr_a = 0.001, # 学习率
                 lr_c = 0.01,
                 gamma = 0.9,  # 折扣率
                 beta = 0.001,  # 正则惩罚系数
                 batch_size=10
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.beta = beta
        self.batch_size = batch_size
        self.ep_rewards = []

        # 定义神经网络架构
        self.actor = ActorNet(self.n_actions, self.n_features)
        self.critic = CriticNet(self.n_features)

        # 采样样本
        self.ep_s, self.ep_r, self.ep_s_, self.ep_a_log_probs, self.ep_a_entropy = [], [], [], [], []
        self.ep_done = []

        # 定义优化器
        self.opt_a = t.optim.Adam(self.actor.parameters(), lr = self.lr_a)
        self.opt_c = t.optim.Adam(self.critic.parameters(), lr = self.lr_c)

        # 定义损失函数形式
        self.c_loss_fn = t.nn.MSELoss()

    def choose_action(self, observation):
        x = t.unsqueeze(t.FloatTensor(observation), 0)

        action_prob = self.actor(x)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()

        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()
        action_selected = action.data.numpy()[0]

        return action_selected, action_log_prob, action_entropy

    def store_transition(self, s, r, done, s_, log_prob, entropy):
        self.ep_s.append(s)
        self.ep_r.append(r)
        self.ep_done.append(done)
        self.ep_s_.append(s_)
        self.ep_a_log_probs.append(log_prob)
        self.ep_a_entropy.append(entropy)

    def learn(self):
        discounted_r = self.discount_reward()

        b_s = t.FloatTensor(self.ep_s)
        b_s_ = t.FloatTensor(self.ep_s_)
        b_r = t.FloatTensor(discounted_r).unsqueeze(dim=1)
        b_r_2 = t.FloatTensor(self.ep_r).unsqueeze(dim=1)
        b_done = t.FloatTensor(self.ep_done).unsqueeze(dim=1)

        # train actor network
        values = self.critic(b_s)
        # td_error = b_r  - values.detach()
        td_error = (b_r_2 + self.gamma * self.critic(b_s_) * (1 - b_done)).detach() - values

        # 计算熵
        entropy = -t.mean(t.cat(self.ep_a_entropy).unsqueeze(dim=1))

        # 计算Actor的目标函数
        a_loss = - t.mean(td_error.detach() * t.cat(self.ep_a_log_probs).unsqueeze(dim=1)) - self.beta * entropy

        self.opt_a.zero_grad()
        a_loss.backward()
        t.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.opt_a.step()

        # train critic network  # 注：critic网络的损失函数放在后边，这种就不用重新计算V(s)值，提高运算效率
        criterion = nn.MSELoss()
        c_loss = td_error.pow(2).mean()

        self.opt_c.zero_grad()
        c_loss.backward()
        self.opt_c.step()

    def clear(self):
        self.ep_s.clear()
        self.ep_r.clear()
        self.ep_s_.clear()
        self.ep_a_log_probs.clear()
        self.ep_a_entropy.clear()
        self.ep_done.clear()

    def discount_reward(self):
        discounted_r = np.zeros_like(self.ep_r)
        final_r = self.ep_r[-1]
        running_add = final_r

        for t in reversed(range(0, len(self.ep_r))):
            running_add = running_add * self.gamma + self.ep_r[t]
            discounted_r[t] = running_add

        return discounted_r

