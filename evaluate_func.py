# -*- coding: UTF-8 -*- #
"""
@filename:evaluate_func.py
@author:201300086
@time:2023-06-30
"""
import numpy as np

def L2_dis(a, b):
    return np.linalg.norm(a - b, ord=2)
    #return np.sqrt(np.sum(np.square(vec1 - vec2)))

def L1_dis(a, b):
    return np.linalg.norm(a - b, ord=1)
    #return np.sum(np.absolute(x-y))

def distance_matrix(X):
    # 直观的距离计算实现方法
    # 首先初始化一个空的距离矩阵D
    D = np.zeros((X.shape[0], X.shape[0]))
    # 循环遍历每一个样本对
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            # 计算样本i和样本j的距离
            D[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
    return D

#留出法
def hold_out(X,Y,train_size=0.8):
    # 划分训练集和测试集
    # 首先打乱样本顺序
    # np.random.seed(0)
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    Y = Y[shuffle_index]
    # 划分训练集和测试集
    train_size = int(X.shape[0] * train_size)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return X_train,Y_train,X_test,Y_test

#k折交叉验证
def k_fold(X,Y,k=5):
    # 首先打乱样本顺序
    # np.random.seed(0)
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    Y = Y[shuffle_index]
    # 划分训练集和测试集
    X_folds = np.array_split(X, k)
    Y_folds = np.array_split(Y, k)
    return X_folds,Y_folds

#自助法采样m次，有放回采样，m为样本总数。用于bagging
def boot_strap(X, Y):
    # 首先打乱样本顺序
    X = np.array(X)
    Y = np.array(Y)
    m = X.shape[0]
    # np.random.seed(0)
    shuffle_index = np.random.permutation(m)
    X = X[shuffle_index]
    Y = Y[shuffle_index]
    # 划分训练集和测试集
    X_train = []
    Y_train = []
    for i in range(m):
        index = np.random.randint(0, m)
        X_train.append(X[index])
        Y_train.append(Y[index])
    return np.array(X_train), np.array(Y_train)
def monte_carlo(X,Y,m=10):
    # 首先打乱样本顺序
    # np.random.seed(0)
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    Y = Y[shuffle_index]
    # 划分训练集和测试集
    X_train = []
    Y_train = []
    for i in range(m):
        X_train.append(X[i])
        Y_train.append(Y[i])
    return X_train,Y_train

if __name__=="__main__":
    #测试留出法
    X=np.random.rand(10,5)
    Y=np.random.rand(10,1)
    X_train,Y_train,X_test,Y_test=hold_out(X,Y)

    #测试k折交叉验证
    X_folds,Y_folds=k_fold(X,Y)
    print(X_folds)
    print(X_folds[1])