import numpy as np


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values     capacity为叶节点的个数
        self.tree = np.zeros(2 * capacity - 1)  # tree定义了一个一维矩阵(只有一行)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        # data记录所有的经验
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        # data_pointer数据指针，指向叶节点
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        # p为数据的优先级，根据树的索引更新优先级
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1  # 数据指针迁移
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0   # 超过容量，数据指针置0

    def update(self, tree_idx, p):
        # 之前的优先级与当前的优先级之差
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2  # //表示取整
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        # 父节点索引从0开始
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            # 叶子节点的左和右索引
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            # 判断是否到达底部(叶子)
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        # 找到数据索引
        data_idx = leaf_idx - self.capacity + 1
        # 返回叶子索引及其对应的优先级 数据transition
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        # 返回根节点的优先级
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # epsilon = 0.01  # small amount to avoid zero priority ---delta+epsilon
    # alpha = 0.6  # [0~1] convert the importance of TD error to priority ---重要性采样因子
    # beta = 0.4  # importance-sampling, from initial value increasing to 1  ---重要性采样因子
    # beta_increment_per_sampling = 0.001
    # abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, epsilon = 0.01, alpha = 0.6, beta = 0.4,
        beta_increment_per_sampling = 0.001, abs_err_upper = 1):
        self.epsilon = epsilon
        self.alpha =alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper     # 确保新存进来的经验有有机会被采样
        self.tree.add(max_p, transition)   # set the max p for new p    # 保证新存进来的都有最高的优先级

    def sample(self, n):
        # 建立采样的索引 记忆 权重
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment 优先级分段
        # 更新beta的值， 采样一次，更新一次
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        # 注：此处total_p实际上是所有|delta_i+epsilon|^alpha的总和
        # 找到最小的概率
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            # 计算当前的采样概率
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
            # b_idx用于后续更新对应数据的优先级使用的索引
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        # 防止0除
        abs_errors += self.epsilon  # convert to abs and avoid 0
        # 不超过1
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)     # 原文中要求clip到[-1,1] reward也要求[-1,1]
        ps = np.power(clipped_errors, self.alpha)
        # 注：更新的优先级实际上是|delta_i+epsilon|^alpha
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)