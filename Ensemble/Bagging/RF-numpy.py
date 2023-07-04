# -*- coding: UTF-8 -*- #
"""
@filename:RF-numpy.py
@author:201300086
@time:2023-07-03
"""

import numpy as np

#建议代码，也算是Begin-End中的一部分
from collections import  Counter
from sklearn.tree import DecisionTreeClassifier
from evaluate_func import boot_strap
from utils import vote

class RandomForestClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        #分类器的数量，默认为10
        self.n_model = n_model
        #用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []
        #用于保存决策树训练时随机选取的列的索引
        self.col_indexs = []


    def fit(self, feature, label):
        '''
        训练模型
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''

        #************* Begin ************#

        for i in range(self.n_model):
            clf = DecisionTreeClassifier(splitter='best')
            feature, label = boot_strap(feature, label)#自助法采样
            #k 的取值一般为 log2(特征数量)
            k=int(np.ceil(np.log2(feature.shape[1])))
            # 从采样到的数据中随机抽取K个特征构成训练集
            index=np.random.permutation(feature.shape[1])[:k]

            self.col_indexs.append(index)
            clf.fit(feature[::,index], label)
            self.models.append(clf)

        #************* End **************#


    def predict(self, feature):
        '''
        :param feature:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        #************* Begin ************#
        #在预测时所用到的特征必须与训练模型时所用到的特征保持一致。
        #例如，第 3 棵决策树在训练时用到了训练集的第 2，5，8 这 3 个特征
        #那么在预测时也要用第 2，5，8 这 3 个特征所组成的测试集传给第 3 棵决策树进行预测
        args = []
        for i,clf in enumerate(self.models):
            result = clf.predict(feature[::,self.col_indexs[i]])
            args.append(result)
        results = vote(*args)
        return results

        #************* End **************#
