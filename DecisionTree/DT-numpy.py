# -*- coding: UTF-8 -*- #
"""
@filename:DT-numpy.py
@author:201300086
@time:2023-07-02
"""
import numpy as np
from performance_measure import ent_n,InfoGain
from copy import deepcopy
from plot_tree_from_dict import createPlot
class DecisionTree(object):
    def __init__(self):
        #决策树模型
        self.tree = {}

    # 获得信息增益最高的特征
    def getBestFeature(self, feature, label):
        max_infogain = 0
        best_feature = 0
        for i in range(len(feature[0])):
            infogain = InfoGain(feature, label, i)
            if infogain > max_infogain:
                max_infogain = infogain
                best_feature = i
        return best_feature
    def createTree(self, feature, label):
        # 样本里都是同一个label没必要继续分叉了
        if len(set(label)) == 1:
            return label[0]
        # 样本中只有一个特征或者所有样本的特征都一样的话就看哪个label的票数高
        if len(feature[0]) == 1 or len(np.unique(feature, axis=0)) == 1:
            vote = {}
            for l in label:
                if l in vote.keys():
                    vote[l] += 1
                else:
                    vote[l] = 1
            max_count = 0
            vote_label = None
            for k, v in vote.items():
                if v > max_count:
                    max_count = v
                    vote_label = k
            return vote_label
        # 根据信息增益拿到特征的索引
        best_feature = self.getBestFeature(feature, label)
        tree = {best_feature: {}}
        f = np.array(feature)
        # 拿到bestfeature的所有特征值
        f_set = set(f[:, best_feature])
        # 构建对应特征值的子样本集sub_feature, sub_label
        for v in f_set:
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][best_feature] == v:
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            # 递归构建决策树
            tree[best_feature][v] = self.createTree(sub_feature, sub_label)
        return tree



    # 计算验证集准确率
    def calc_acc_val(self, the_tree, val_feature, val_label):
        result = []
        for f in val_feature:
            result.append(self.classify(the_tree, f))
        result = np.array(result)
        return np.mean(result == val_label)

    # 后剪枝
    def post_cut(self, val_feature, val_label):
        # 拿到非叶子节点的数量
        def get_non_leaf_node_count(tree):
            non_leaf_node_path = []

            def dfs(tree, path, all_path):
                for k in tree.keys():
                    if isinstance(tree[k], dict):
                        path.append(k)
                        dfs(tree[k], path, all_path)
                        if len(path) > 0:
                            path.pop()
                    else:
                        all_path.append(path[:])

            dfs(tree, [], non_leaf_node_path)
            unique_non_leaf_node = []
            for path in non_leaf_node_path:
                isFind = False
                for p in unique_non_leaf_node:
                    if path == p:
                        isFind = True
                        break
                if not isFind:
                    unique_non_leaf_node.append(path)
            return len(unique_non_leaf_node)

        # 拿到树中深度最深的从根节点到非叶子节点的路径
        def get_the_most_deep_path(tree):
            non_leaf_node_path = []

            def dfs(tree, path, all_path):
                for k in tree.keys():
                    if isinstance(tree[k], dict):
                        path.append(k)
                        dfs(tree[k], path, all_path)
                        if len(path) > 0:
                            path.pop()
                    else:
                        all_path.append(path[:])

            dfs(tree, [], non_leaf_node_path)
            max_depth = 0
            result = None
            for path in non_leaf_node_path:
                if len(path) > max_depth:
                    max_depth = len(path)
                    result = path
            return result

        # 剪枝
        def set_vote_label(tree, path, label):
            for i in range(len(path) - 1):
                tree = tree[path[i]]
            tree[path[len(path) - 1]] = vote_label

        acc_before_cut = self.calc_acc_val(self.tree, val_feature, val_label)
        # 遍历所有非叶子节点
        for _ in range(get_non_leaf_node_count(self.tree)):
            path = get_the_most_deep_path(self.tree)
            # 备份树
            tree = deepcopy(self.tree)
            step = deepcopy(tree)
            # 跟着路径走
            for k in path:
                step = step[k]
            # 叶子节点中票数最多的标签
            vote_label = sorted(step.items(), key=lambda item: item[1], reverse=True)[0][0]
            # 在备份的树上剪枝
            set_vote_label(tree, path, vote_label)
            acc_after_cut = self.calc_acc_val(tree, val_feature, val_label)
            # 验证集准确率高于0.9才剪枝
            if acc_after_cut > acc_before_cut:
                set_vote_label(self.tree, path, vote_label)
                acc_before_cut = acc_after_cut

    def fit(self, train_feature, train_label, val_feature, val_label):
        '''
        :param train_feature:训练集数据，类型为ndarray
        :param train_label:训练集标签，类型为ndarray
        :param val_feature:验证集数据，类型为ndarray
        :param val_label:验证集标签，类型为ndarray
        :return: None
        '''
        #************* Begin ************#
        self.tree=self.createTree(train_feature, train_label)
        self.post_cut(val_feature, val_label)
        print(self.tree)
        #************* End **************#

    # 单个样本分类(核心，易错)
    def classify(self,tree, feature):
        if not isinstance(tree, dict):
            return tree
        key, t_value = list(tree.items())[0]
        if isinstance(t_value, dict):#需要再判断一次
            return self.classify(tree[key][feature[key]], feature)
        else:
            return t_value

    #批量样本分类
    def predict(self, features):
        '''
        :param features:测试集数据，类型为ndarray
        :return:预测结果，如np.array([0, 1, 2, 2, 1, 0])
        '''
        result = []
        for f in features:
            result.append(self.classify(self.tree, f))
        return np.array(result)

