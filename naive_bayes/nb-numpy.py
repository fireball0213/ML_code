# -*- coding: UTF-8 -*- #
"""
@filename:nb-numpy.py
@author:201300086
@time:2023-06-30
"""
import numpy as np
#准确率很低，不知道bug在哪里

class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}  # 先验
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}  # 似然

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''
        # print(label)
        # print(np.bincount(label))
        bin_lst = np.bincount(label)
        label_num = sum(bin_lst)
        for i in range(len(bin_lst)):
            self.label_prob[i] = bin_lst[i] / label_num
        print(self.label_prob)
        # print(feature.T)
        # print(feature)
        for i in range(len(bin_lst)):
            l_dic = {}
            for j in range(len(feature[0])):  # 3列
                col_dic = {}
                # 筛选标签为i的行
                tmp_f = []
                for p in range(len(feature)):
                    if label[p] == i:
                        tmp_f.append(feature[p])
                col_data = np.array(tmp_f).T[j]
                col_data_count = np.bincount(col_data)
                for k in range(1, len(col_data_count)):
                    col_dic[k] = col_data_count[k] / sum(col_data_count)
                # print(col_data)
                # print(col_data_count)
                # print(col_dic)
                l_dic[j] = col_dic
            # print(l_dic)
            self.condition_prob[i] = l_dic
        print(self.condition_prob)


    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        pred = []
        for key, value in self.label_prob.items():
            print(key, value)
            prob_lst = []
            for data in feature:
                tmp = value
                for j in range(len(data)):
                    # print(data,key,j,data[j],self.condition_prob[key][j][data[j]])
                    tmp = tmp + self.condition_prob[key][j][data[j]]
                prob_lst.append(tmp)
            pred.append(np.array(prob_lst))
        pred = np.array(pred).T
        print(pred)
        result = []
        for i in pred:
            print(i)
            if (i[0] < i[1]):
                result.append(0)
            else:
                result.append(1)
        return result
