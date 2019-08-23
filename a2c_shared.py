import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class ActorCriticNet(nn.Module):
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
        self.fc4 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        action_dist = F.softmax(self.fc3(out), dim=1)
        value = self.fc4(out)

        return action_dist, value

class A2C_Shared(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 lr = 0.001, # 学习率
                 gamma = 0.9,  # 折扣率
                 beta = 0.001, # 正则惩罚系数
                 zeta = 0.5,    # Critic梯度权重系数
                 batch_size = 10
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = gamma
        self.beta = beta
        self.zeta = zeta
        self.batch_size = batch_size
        self.policy_loss = []
        self.value_loss = []
        self.entropy_loss = []
        self.kl_div = []
        self.ep_rewards = []

        # 定义神经网络架构
        self.ac_net = ActorCriticNet(self.n_actions, self.n_features)

        # 采样样本
        self.ep_s, self.ep_r, self.ep_s_, self.ep_a_log_probs, self.ep_a_entropy = [], [], [], [], []
        self.ep_v = []
        self.ep_a_probs = []
        self.ep_done = []

        # 定义优化器
        self.optim = t.optim.Adam(self.ac_net.parameters(), lr = self.lr)


        # 定义损失函数形式
        self.c_loss_fn = t.nn.MSELoss()

    def choose_action(self, observation):
        x = t.unsqueeze(t.FloatTensor(observation), 0)

        action_prob, value = self.ac_net(x)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()

        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()
        action_selected = action.data.numpy()[0]

        return action_prob, action_selected, action_log_prob, action_entropy, value

    def store_transition(self, s, r, done, s_, prob, log_prob, entropy, value):
        self.ep_s.append(s)
        self.ep_r.append(r)
        self.ep_done.append(done)
        self.ep_s_.append(s_)
        self.ep_a_log_probs.append(log_prob)
        self.ep_a_entropy.append(entropy)
        self.ep_v.append(value)
        self.ep_a_probs.append(prob)

    def learn(self):
        discounted_r = self.discount_reward()

        b_s = t.FloatTensor(self.ep_s)
        b_s_ = t.FloatTensor(self.ep_s_)
        b_r = t.FloatTensor(discounted_r).unsqueeze(dim=1)
        b_r_2 = t.FloatTensor(self.ep_r).unsqueeze(dim=1)
        b_done = t.FloatTensor(self.ep_done).unsqueeze(dim=1)

        # train network
        values = t.cat(self.ep_v)
        # td_error = b_r  - values

        # Fig3
        td_error = (b_r_2 + self.gamma * self.ac_net(b_s_)[1] * (1 - b_done)).detach() - values

        # 计算熵
        entropy = - self.beta * t.mean(t.cat(self.ep_a_entropy).unsqueeze(dim=1))

        # 计算目标函数
        a_loss = - t.mean(td_error * t.cat(self.ep_a_log_probs).unsqueeze(dim=1))
        c_loss = self.zeta * td_error.pow(2).mean()

        # Append values
        self.policy_loss.append(a_loss)
        self.value_loss.append(c_loss)
        self.entropy_loss.append(entropy)

        ac_loss = a_loss + 0.5 * c_loss - entropy

        loss1 = a_loss - entropy
        loss2 = c_loss
        self.optim.zero_grad()
        # ac_loss.backward(retain_graph = False)  # Fig1
        # ac_loss.backward(retain_graph = True)   # Fig2
        # Fig5
        loss1.backward(retain_graph = True)
        loss2.backward()
        t.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
        self.optim.step()

    def clear(self):
        self.ep_s.clear()
        self.ep_r.clear()
        self.ep_s_.clear()
        self.ep_a_log_probs.clear()
        self.ep_a_entropy.clear()
        self.ep_v.clear()
        self.ep_a_probs.clear()
        self.ep_done.clear()

    def discount_reward(self):
        discounted_r = np.zeros_like(self.ep_r)
        final_r = self.ep_r[-1]
        running_add = final_r

        for t in reversed(range(0, len(self.ep_r))):
            running_add = running_add * self.gamma + self.ep_r[t]
            discounted_r[t] = running_add

        return discounted_r
    def kl_calc(self):
        b_s = t.FloatTensor(self.ep_s)

        new_action_probs= self.ac_net(b_s)[0].detach()
        old_action_probs = t.cat(self.ep_a_probs).detach()
        kl = -t.sum(old_action_probs * t.log(new_action_probs / old_action_probs))
        self.kl_div.append(kl.numpy())

    def plot_results(self):
        avg_rewards = [np.mean(self.ep_rewards[i:i + self.batch_size])
                       if i > self.batch_size
                       else np.mean(self.ep_rewards[:i + 1]) for i in range(len(self.ep_rewards))]


        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2)
        ax0 = plt.subplot(gs[0, :])
        ax0.plot(self.ep_rewards)
        ax0.plot(avg_rewards)
        ax0.set_xlabel('Episode')
        plt.title('Rewards')

        ax1 = plt.subplot(gs[1, 0])
        ax1.plot(self.policy_loss)
        plt.title('Policy Loss')
        plt.xlabel('Update Number')

        ax2 = plt.subplot(gs[1, 1])
        ax2.plot(self.entropy_loss)
        plt.title('Entropy Loss')
        plt.xlabel('Update Number')

        ax3 = plt.subplot(gs[2, 0])
        ax3.plot(self.value_loss)
        plt.title('Value Loss')
        plt.xlabel('Update Number')

        ax4 = plt.subplot(gs[2, 1])
        ax4.plot(self.kl_div)
        plt.title('KL Divergence')
        plt.xlabel('Update Number')

        plt.tight_layout()
        # plt.show()
        plt.savefig('figure_1.png')
        plt.close()
