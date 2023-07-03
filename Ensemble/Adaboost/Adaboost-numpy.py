# encoding=utf8
import numpy as np


# adaboost算法
class AdaBoost:
    '''
    input:n_estimators(int):迭代轮数
          learning_rate(float):弱分类器权重缩减系数
    '''

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape
        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化weights
        self.weights = [1.0 / self.M] * self.M
        # G(x)系数 alpha
        self.alpha = []

    def _G(self, features, labels, weights):
        m = len(features)
        error = 100000.0  # 初始错误率设为无穷大
        best_v = 0.0
        # 单维特征
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        # 寻找可能的阈值
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i
            if v not in features:
                # 误分类计算
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])

                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                # 选取误差小的作为当前的分类器
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array

    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    def _Z(self, weights, a, clf):
        return sum([weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)])

    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z

    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        self.init_args(X, y)

        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)

                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j

                if best_clf_error == 0:
                    break

            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))
            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)
            # 权值更新
            self._w(a, clf_result, Z)

    def predict(self, data):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = data[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        # sign
        return 1 if result > 0 else -1

# adaboost算法
# class AdaBoost:
#     '''
#     input:n_estimators(int):迭代轮数
#           learning_rate(float):弱分类器权重缩减系数
#     '''
#     def __init__(self, n_estimators=50, learning_rate=1.0):
#         self.clf_num = n_estimators
#         self.learning_rate = learning_rate
#     def init_args(self, datasets, labels):
#         self.X = datasets
#         self.Y = labels
#         self.M, self.N = datasets.shape
#         # 弱分类器数目和集合
#         self.clf_sets = []
#         # 初始化weights
#         self.weights = [1.0/self.M]*self.M
#         # G(x)系数 alpha
#         self.alpha = []
#         print(self.X,self.Y,self.M,self.N)
#     #********* Begin *********#
#     def _G(self, features, labels, weights):
#         '''
#         input:features(ndarray):数据特征
#               labels(ndarray):数据标签
#               weights(ndarray):样本权重系数
#         '''
#         print(features, labels, weights)
#         # e= np.sum(labels!=features)
#         return e
#
#     # 计算alpha
#     def _alpha(self, error):
#         # return 0.5*np.log((1-error)/error)
#
#     # 规范化因子
#     def _Z(self, weights, a, clf):
#         # return np.sum(weights.dot(np.exp(-np.dot(a,clf))))
#
#     # 权值更新
#     def _w(self, a, clf, Z):
#         for i in range(self.M):
#
#
#     # G(x)的线性组合
#     def G(self, x, v, direct):
#
#     def fit(self, X, y):
#         '''
#         X(ndarray):训练数据
#         y(ndarray):训练标签
#         '''
#         self.init_args(features, labels)
#
#             # 计算G(x)系数a
#
#             # 记录分类器
#
#             # 规范化因子
#
#             # 权值更新
#
#     def predict(self, data):
#         '''
#         input:data(ndarray):单个样本
#         output:预测为正样本返回+1，负样本返回-1
#         '''
#
#     #********* End *********#
