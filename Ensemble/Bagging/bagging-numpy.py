# -*- coding: UTF-8 -*- #
"""
@filename:bagging-numpy.py
@author:201300086
@time:2023-07-03
"""

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from evaluate_func import boot_strap
from utils import vote

class BaggingClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        # 分类器的数量，默认为10
        self.n_model = n_model
        # 用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []

    def fit(self, feature, label):
        '''
        训练模型，请记得将模型保存至self.models
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''

        # ************* Begin ************#
        for i in range(self.n_model):
            clf = DecisionTreeClassifier(splitter='best')  # splitter='best''random'
            feature, label = boot_strap(feature, label)#自助法采样
            clf.fit(feature, label)
            self.models.append(clf)

        # ************* End **************#

    def predict(self, feature):
        '''
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        # ************* Begin ************#

        args = []
        for clf in self.models:
            result = clf.predict(feature)
            args.append(result)
        results = vote(*args)
        return results

        # ************* End **************#
