# -*- coding: UTF-8 -*- #
"""
@filename:LogisticR-OvR-numpy.py
@author:201300086
@time:2023-07-02
"""
import numpy as np


# 逻辑回归
class tiny_logistic_regression(object):
    def __init__(self):
        # W
        self.coef_ = None
        # b
        self.intercept_ = None
        # 所有的W和b
        self._theta = None
        # 01到标签的映射
        self.label_map = {}

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    # 训练
    def fit(self, train_datas, train_labels, learning_rate=1e-4, n_iters=1e3):
        # loss
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')

        # 算theta对loss的偏导
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

        # 批量梯度下降
        def gradient_descent(X_b, y, initial_theta, leraning_rate, n_iters=1e2, epsilon=1e-6):
            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - leraning_rate * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(train_datas), 1)), train_datas])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, train_labels, initial_theta, learning_rate, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    # 预测X中每个样本label为1的概率
    def predict_proba(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return self._sigmoid(X_b.dot(self._theta))

    # 预测
    def predict(self, X):
        proba = self.predict_proba(X)
        result = np.array(proba >= 0.5, dtype='int')
        return result


class OvR(object):
    def __init__(self):
        # 用于保存训练时各种模型的list
        self.models = []
        # 用于保存models中对应的正例的真实标签
        # 例如第1个模型的正例是2，则real_label[0]=2
        self.real_label = []

    def fit(self, train_datas, train_labels):
        '''
        OvO的训练阶段，将模型保存到self.models中
        :param train_datas: 训练集数据，类型为ndarray
        :param train_labels: 训练集标签，类型为ndarray，shape为(-1,)
        :return:None
        '''
        # ********* Begin *********#

        X = train_datas
        for i in range(3):
            model = tiny_logistic_regression()
            real_label = i
            Y = train_labels.copy()#注意必须深拷贝
            #先统一换到其他值，再赋值，否则破坏原数据！！！
            Y[Y != i] = 3
            Y[Y == i] = 4
            Y[Y != 4] = 0
            Y[Y == 4] = 1
            model.fit(X, Y)  # 第i类样本为正例
            self.models.append(model)
            self.real_label.append(real_label)

        # ********* End *********#

    def predict(self, test_datas):
        '''
        OvO的预测阶段
        :param test_datas:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray
        '''
        # ********* Begin *********#
        result = []
        for m in self.models:
            result.append(m.predict_proba(test_datas))#使用概率输出
        # result = np.array(result).reshape(len(self.models), len(test_datas))
        result = np.argmax(result, axis=0)#求每列最大概率所在行的index
        return result
        # ********* End *********#