# -*- coding: UTF-8 -*- #
"""
@filename:KNN-numpy.py
@author:201300086
@time:2023-06-30
"""
#encoding=utf8
import numpy as np

class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None


    def fit(self, feature, label):
        '''
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        '''

        #********* Begin *********#
        self.train_feature = feature
        self.train_label = label
        #********* End *********#


    def predict(self, feature):
        '''
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        '''

        #********* Begin *********#
        def L2(a,b):
            return np.linalg.norm(a-b,ord=2)
        def vote(index_lst):
            label_lst=[]
            for i in index_lst:
                label_lst.append(self.train_label[i])
            # print(label_lst)
            # print(np.bincount(label_lst) )
            # print(np.argmax(np.bincount(label_lst) ))
            pre=label_lst[np.argmax(np.bincount(label_lst) )]
            # print('pred:',pre)
            return pre

        pred=[]
        for i in range(len(feature)):
            feat_lst=[]
            f=self.train_feature
            for j in range(len(f)):
                feat_lst.append(L2(feature[i],f[j]))
            k_index_feat_lst=np.argsort(feat_lst)[: self.k]
            # print(k_index_feat_lst)
            # print(feat_lst)
            pred.append(vote(k_index_feat_lst))

        return np.array(pred)



        #********* End *********#
