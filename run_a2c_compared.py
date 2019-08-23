import numpy as np
from a2c_seperated import A2C_Seperated
from a2c_shared import A2C_Shared
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

MAX_EPISODE = 2000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 200   # maximum time step in one episode
RENDER = False  # rendering wastes time

np.random.seed(2)

env = gym.make('CartPole-v0')

env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n



def run(a2c_learner):
    for i_episode in tqdm(range(MAX_EPISODE)):
        s = env.reset()

        # while True:
        for i_step in range(MAX_EP_STEPS):
            if RENDER: env.render()

            action_prob, action, action_log_prob, action_entropy, value = a2c_learner.choose_action(s)

            s_, r, done, info = env.step(action)

            a2c_learner.store_transition(s, r, done, s_, action_prob, action_log_prob, action_entropy, value)

            if done or i_step == MAX_EP_STEPS - 1:
                ep_rs_sum = sum(a2c_learner.ep_r)
                a2c_learner.ep_rewards.append(ep_rs_sum)

                # if 'running_reward' not in globals():
                #     running_reward = ep_rs_sum
                # else:
                #     running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = False     # rendering
                # print("episode:", i_episode, "  reward:", int(running_reward))

                a2c_learner.learn()

                # a2c_learner.kl_calc()

                a2c_learner.clear()

                break

            s = s_

    # a2c_learner.plot_results()

def run2(a2c_learner):
    for i_episode in tqdm(range(MAX_EPISODE)):
        s = env.reset()

        # while True:
        for i_step in range(MAX_EP_STEPS):
            if RENDER: env.render()

            action, action_log_prob, action_entropy = a2c_learner.choose_action(s)

            s_, r, done, info = env.step(action)

            a2c_learner.store_transition(s, r, done, s_, action_log_prob, action_entropy)

            if done or i_step == MAX_EP_STEPS - 1:
                ep_rs_sum = sum(a2c_learner.ep_r)
                a2c_learner.ep_rewards.append(ep_rs_sum)

                # if 'running_reward' not in globals():
                #     running_reward = ep_rs_sum
                # else:
                #     running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = False     # rendering
                # print("episode:", i_episode, "  reward:", int(running_reward))

                a2c_learner.learn()

                # a2c_learner.kl_calc()

                a2c_learner.clear()

                break

            s = s_

def plot_results(learner1, learner2):
    avg_rewards1 = [np.mean(learner1.ep_rewards[i:i + learner1.batch_size])
                   if i > learner1.batch_size
                   else np.mean(learner1.ep_rewards[:i + 1]) for i in range(len(learner1.ep_rewards))]

    avg_rewards2 = [np.mean(learner2.ep_rewards[i:i + learner2.batch_size])
                   if i > learner2.batch_size
                   else np.mean(learner2.ep_rewards[:i + 1]) for i in range(len(learner2.ep_rewards))]

    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 2)
    ax0 = plt.subplot(gs[0, :])
    ax0.plot(learner1.ep_rewards, label='shared network(rewards)')
    ax0.plot(avg_rewards1,label='shared network(avg rewards)')
    ax0.plot(learner2.ep_rewards,label='seperated network(rewards)')
    ax0.plot(avg_rewards2,label='seperated network(avg rewards)')
    ax0.set_xlabel('Episode')
    plt.title('Rewards')

    # ax1 = plt.subplot(gs[1, 0])
    # ax1.plot(self.policy_loss)
    # plt.title('Policy Loss')
    # plt.xlabel('Update Number')
    #
    # ax2 = plt.subplot(gs[1, 1])
    # ax2.plot(self.entropy_loss)
    # plt.title('Entropy Loss')
    # plt.xlabel('Update Number')
    #
    # ax3 = plt.subplot(gs[2, 0])
    # ax3.plot(self.value_loss)
    # plt.title('Value Loss')
    # plt.xlabel('Update Number')
    #
    # ax4 = plt.subplot(gs[2, 1])
    # ax4.plot(self.kl_div)
    # plt.title('KL Divergence')
    # plt.xlabel('Update Number')

    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('figure_01.png')
    plt.close()

a2c_learner1 = A2C_Shared(N_A, N_F)
a2c_learner2 = A2C_Seperated(N_A, N_F)
run(a2c_learner1)
run2(a2c_learner2)

plot_results(a2c_learner1, a2c_learner2)



