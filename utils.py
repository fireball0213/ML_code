# -*- coding: UTF-8 -*- #
"""
@filename:utils.py
@author:201300086
@time:2023-07-02
"""
import numpy as np
from collections import Counter

#OvO多分类与集成学习中的投票函数
def vote(*args):
    """
    :param args: n个numpy结果列表
    :return: 每一维的投票最终结果构成的数组
    """
    result = []
    #以转置方式计算temp，而不是循环
    # 使用np.bincount实现投票

    for i in range(len(args[0])):
        temp = []
        for j in range(len(args)):
            temp.append(args[j][i])
        result.append(np.argmax(np.bincount(np.array(temp))))
    result = np.array(result)
    # result = np.argmax(np.bincount(np.array(args).T))#太简单的写法，会报错？
    #使用Counter类实现投票
    # result = np.array([Counter(temp).most_common(1)[0][0] for temp in np.array(args).T])

    return result



if __name__=="__main__":
    #vote函数测试，输入为n个np.array([1, 2, 3, 4, 5])，输出为最终结果，n=100
    result = []
    for i in range(100):
        result.append(np.array([1, 2, 3, 4, 5]))
    result = vote(*result)
    print(result)
    print(~np.array([1,0,1,2,-1]))


