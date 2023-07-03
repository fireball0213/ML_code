# -*- coding: UTF-8 -*- #
"""
@filename:Perceptron.py
@author:201300086
@time:2023-07-01
"""
# encoding=utf8
import numpy as np


# 构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.] * data.shape[1])
        self.b = np.array([1.])
        # ********* Begin *********#
        epoch = 0
        while (epoch < self.max_iter):
            for i in range(len(data)):
                x = data[i]
                y = label[i]
                if y * np.sum(self.w * x + self.b) <= 0:
                    self.w = self.w + self.lr * y * x
                    self.b = self.b + self.lr * y
            epoch += 1
        # ********* End *********#

    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        # ********* Begin *********#
        pred = []
        for d in data:#不遍历就变成全局均值了
            y_hat = np.sum(self.w * d + self.b)
            pred.append(np.sign(y_hat))
        pred = np.array(pred)
        # ********* End *********#
        return pred

