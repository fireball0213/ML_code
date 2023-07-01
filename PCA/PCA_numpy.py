# -*- coding: UTF-8 -*- #
"""
@filename:PCA_numpy.py
@author:201300086
@time:2023-07-01
"""
#使用numpy不使用sklearn的PCA函数
import numpy as np
"""
X每列是特征，每行是样本
axis=0表示按列求均值
计算协方差需要转置，因为cov函数的输入希望是行代表特征，列代表数据的矩阵，所以要转置
根据A的顺序选B这种，一律用argsort ，加-号表示从大到小排序
特征向量矩阵的每列是一个特征向量，不是每行
最后是X*P，不是P*X
"""
def PCA(X, n_components):#X每列是特征，每行是样本
    # 首先对原始数据零均值化
    X = X - np.mean(X, axis=0)#axis=0表示按列求均值
    # 计算协方差矩阵
    cov = np.cov(X, rowvar=False)#相当于X转置了，rowvar=False表示每一列代表一个特征
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # 对特征向量按照特征值大小进行排序
    index = np.argsort(-eigenvalues)#根据A的顺序选B这种，一律用argsort  #argsort默认是从小到大排序，加-号表示从大到小排序
    # 选取前n_components个特征向量
    P = eigenvectors[:, index][:, :n_components]#每列是一个特征向量，不是每行
    # 将数据转换到新的低维空间
    X_pca = np.dot(X, P)
    return X_pca
