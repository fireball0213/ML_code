# -*- coding: UTF-8 -*- #
"""
@filename:LDA-numpy.py
@author:201300086
@time:2023-07-01
"""
##使用numpy实现LDA
import numpy as np
def lda(X, y):
    '''
    input:X(ndarray):待处理数据，维度为m*n，m为样本数，n为特征数
          y(ndarray):待处理数据标签，标签分别为0和1
    '''
    #根据y把X划分成两类
    X0 = X[y==0]
    X1 = X[y==1]
    #计算两类数据的均值向量
    u0 = np.mean(X0, axis=0)
    u1 = np.mean(X1, axis=0)
    #计算两类数据的协方差矩阵
    # cov0 = np.cov(X0.T)
    # cov1 = np.cov(X1.T)
    cov0 = np.dot((X0 - u0).T, (X0 - u0))
    cov1 = np.dot((X1 - u1).T, (X1 - u1))
    #计算总的协方差矩阵
    cov = cov0 + cov1
    #计算投影方向
    w = np.dot(np.linalg.inv(cov), u0-u1)
    #计算投影后的数据，维度为m*1
    X0_new = np.dot(X0, w)
    X1_new = np.dot(X1, w)
    return X0_new, X1_new, w