import numpy as np

# 逻辑回归
class tiny_logistic_regression(object):
    def __init__(self):
        #W
        self.coef_ = None
        #b
        self.intercept_ = None
        #所有的W和b
        self._theta = None
        #01到标签的映射
        self.label_map = {}


    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))


    #训练，train_labels中的值可以为任意数值
    def fit(self, train_datas, train_labels, learning_rate=1e-4, n_iters=1e3):
        #loss
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        # 算theta对loss的偏导
        def dJ(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            return np.dot(X_b.T,y_hat - y) / len(y)

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

        unique_labels = list(set(train_labels))
        labels = np.array(train_labels)

        # 将标签映射成0，1
        self.label_map[0] = unique_labels[0]
        self.label_map[1] = unique_labels[1]

        for i in range(len(train_labels)):
            if train_labels[i] == self.label_map[0]:
                labels[i] = 0
            else:
                labels[i] = 1

        X_b = np.hstack([np.ones((len(train_datas), 1)), train_datas])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, labels, initial_theta, learning_rate, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    #预测X中每个样本label为1的概率
    def predict_proba(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return self._sigmoid(X_b.dot(self._theta))

    #预测
    def predict(self, X):
        proba = self.predict_proba(X)
        result = np.array(proba >= 0.5, dtype='int')
        # 将0，1映射成标签
        for i in range(len(result)):
            if result[i] == 0:
                result[i] = self.label_map[0]
            else:
                result[i] = self.label_map[1]
        return result



class OvO(object):
    def __init__(self):
        # 用于保存训练时各种模型的list
        self.models = []


    def fit(self, train_datas, train_labels):
        '''
        OvO的训练阶段，将模型保存到self.models中
        :param train_datas: 训练集数据，类型为ndarray
        :param train_labels: 训练集标签，标签值为0,1,2之类的整数，类型为ndarray，shape为(-1,)
        :return:None
        '''

        #********* Begin *********#
        X=train_datas
        Y=train_labels
        for i in range(3):
            model=tiny_logistic_regression()
            model.fit(X[Y!=i], Y[Y!=i])#不包含第i类样本的所有数据
            self.models.append(model)

        #********* End *********#


    def predict(self, test_datas):
        '''
        OvO的预测阶段
        :param test_datas:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray
        '''

        #********* Begin *********#
        def vote(*args):
            result = []
            for i in range(len(args[0])):
                temp = []
                for j in range(len(args)):
                    temp.append(args[j][i])
                result.append(np.argmax(np.bincount(temp)))
            result = np.array(result)
            return result

        #标签已映射成0，1，2
        result0= self.models[0].predict(test_datas)#第1类和第2类的分类结果
        result1= self.models[1].predict(test_datas)#第0类和第2类的分类结果
        result2= self.models[2].predict(test_datas)#第0类和第1类的分类结果
        return vote(result0, result1, result2)

        #********* End *********#




