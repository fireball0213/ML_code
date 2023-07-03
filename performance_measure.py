# -*- coding: UTF-8 -*- #
"""
@filename:performance_measure.py
@author:201300086
@time:2023-06-30
"""
import numpy as np
def acc(pred, label):
    return np.sum(pred==label)/len(label)

#计算MSE
def MSE(pred, label):
    return np.sum(np.square(pred-label))/len(label)
    #return np.mean((pred-label)**2)

def r2_score(y_predict,y_test):
    a=MSE(y_predict,y_test)
    b=MSE(np.mean(y_test),y_test)
    return 1-a/b

# 实现计算percision， recall和F1 score的函数
def p_r_f1(Y_pred, Y_gt):

    # 法一：严格按照公式
    # TP,FP,FN=0,0,0
    # for i in range(len(Y_pred)):
    #     if Y_pred[i]==1 and Y_gt[i]==1:
    #         TP+=1
    #     elif Y_pred[i]==1 and Y_gt[i]==0:
    #         FP+=1
    #     elif Y_pred[i]==0 and Y_gt[i]==1:
    #         FN+=1
    # precision=TP/(TP+FP)
    # recall=TP/(TP+FN)
    # f1 = 2 * precision * recall / (precision + recall)

    # 法二：更好理解的查准率和查全率：
    TP = np.sum((Y_pred == 1) & (Y_gt == 1))
    P = np.sum(Y_gt == 1)
    precision = TP / np.sum(Y_pred == 1)  # 在预测为1里面有多少是准的
    recall = TP / P  # 在真实为1里面有多少是预测对的
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def ent_2(p):
    if p == 1 or p == 0:
        return 0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)
    #看清楚是ln还是log2
    # return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def ent_n(label):
    result = 0
    for l in set(label):#比用bincount更通用
        count = np.sum(label == l)
        p = count / len(label)
        result -= p * np.log2(p)
    return result

def ent_gini(p):
    return 1-p**2-(1-p)**2

def InfoGain(feature, label, index):
    '''
    计算信息增益,信息增益率,基尼指数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益,信息增益率,基尼指数，类型float
    '''

    f=feature[::,index]

    #计算标记的分布
    p=np.bincount(label)[0]/len(label)

    #计算条件熵：特征的每个值关于标记的信息熵
    #简化步骤f00=len(f[label==0][f[label==0]==0])
    f00 = np.sum((f == 0) & (label == 0))
    f01 = np.sum((f == 0) & (label == 1))
    f10 = np.sum((f == 1) & (label == 0))
    f11 = np.sum((f == 1) & (label == 1))
    ent0 = ent_2(f00 / (f00 + f01))
    ent1 = ent_2(f10 / (f10 + f11))
    ent0_gini = ent_gini(f00 / (f00 + f01))
    ent1_gini = ent_gini(f10 / (f10 + f11))

    #计算特征的信息增益
    Gain=ent_2(p)-((f00+f01)/len(label))*ent0-((f10+f11)/len(label))*ent1

    #计算特征的信息增益率
    IV = ent_2((f00 + f01) / len(label))
    Gain_ratio = Gain / IV

    # 计算特征的基尼系数
    Gini = ((f00 + f01) / len(label)) * ent0_gini + ((f10 + f11) / len(label)) * ent1_gini

    return Gain, Gain_ratio, Gini
