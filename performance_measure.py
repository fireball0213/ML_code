# -*- coding: UTF-8 -*- #
"""
@filename:performance_measure.py
@author:201300086
@time:2023-06-30
"""
import numpy as np
def acc(pred, label):
    pred=np.array(pred)
    label=np.array(label)
    return np.sum(pred==label)/len(label)


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