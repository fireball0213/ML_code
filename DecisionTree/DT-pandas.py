from math import log
import pandas as pd
from plot_tree_from_dict import createPlot


# 计算信息熵或基尼  （DataFrame）
def Ent(data, flag):  # flag=1信息熵，flag=2基尼指数，flag=3缺失值
    labels = {}
    data = data.reset_index(drop=True)
    if (flag == 1):
        names = data[data.columns[-1]]  # 依据公式求某列特征的熵 目标变量作为概率依据
        n = len(names)
        for i, j in names.value_counts().items():
            labels[i] = j
        shang = 0
        for i in labels:  # 利用循环求熵
            pi = labels[i] / n
            shang -= pi * log(pi, 2)
    elif (flag == 2):
        names = data[data.columns[-1]]
        n = len(names)

        for i, j in names.value_counts().items():
            labels[i] = j
        shang = 1
        for i in labels:
            pi = labels[i] / n
            shang -= pi ** 2
    elif (flag == 3):  # 包含权重计算，最后一列权
        names = data[data.columns[-2]]
        weight = data[data.columns[-1]]
        shang = 0
        label_sum = 0
        for i, j in names.value_counts().items():
            labels[i] = 0  # 初始化
        for m in range(len(weight)):
            labels[names.loc[m]] += weight.loc[m]
            label_sum += weight.loc[m]
        for i in labels:
            pi = labels[i] / label_sum
            shang -= pi * log(pi, 2)
    return shang


def split_dataset(data, feature, feature_value):
    """
    Split the dataset based on the feature and its value.
    """
    return data[data[feature] == feature_value].drop(columns=feature)


def delete_empty_dataset(data, feature):
    """
    Delete the rows with missing values in the specified feature.
    """
    return data[data[feature] != -1].reset_index(drop=True)

# 划分数据集  （DataFrame,特征列名,该列某个特征值）
def splitdataSet(data, feature, feature_value):
    recvdata = []
    n = len(data)
    # print(data)
    for i in range(n):  # 如果该行的这个特征值==循环到的这个特征值，去掉该特征加入返回列表
        if (data.iloc[[i], :][feature].values[0] == feature_value):
            temp = data.iloc[[i], :]
            k = temp.index.values[0]
            temp_t = temp.loc[k]
            tem = temp_t.drop(feature)  # 删除目标属性
            recvdata.append(tem)
    recvDF = pd.DataFrame(recvdata)  # 将满足条件的所有行定义为DataFrame
    # print(recvDF)
    return recvDF


def deleteemptydataSet(data, feature):  # 缺失值划分属性时删去空值行，并更改权重
    recvdata = []
    n = len(data)
    for i in range(n):
        if (data.iloc[[i], :][feature].values[0] != -1):  # 非空值
            temp = data.iloc[[i], :]  # 匹配到的这一行
            k = temp.index.values[0]  # 匹配到的行号
            recvdata.append(temp.loc[k])
    recvDF = pd.DataFrame(recvdata)  # 将满足条件的所有行定义为DataFrame
    recvDF = recvDF.reset_index(drop=True)  # index重排
    # print(recvDF)
    return recvDF


node_num = 1


# 得出最好的特征名，用来划分数据集 （DataFrame）
def splitbest(data, flag):
    global node_num
    if (flag == 3):
        nameFeatures = data.drop(['w'], axis=1).columns
    else:
        nameFeatures = data.columns
    baseEntropy = Ent(data, flag)  # 原始max信息熵
    bestGain = 0.0  # 初始化最好信息增益
    bestGini = 1.0
    bestFeature = -1  # 初始化最好的特征名
    print("node ", node_num, ": ", end=" ")
    for Feature in nameFeatures[:-1]:  # 循环所有属性
        if (flag == 1):
            uniquevalue = set(data[Feature])  # 该特征的所有唯一值
            newEntropy = 0.0  # 中间熵
            if (bestFeature == -1):  # 初始化错误
                bestFeature = Feature
            for value in uniquevalue:
                subdata = splitdataSet(data, Feature, value)
                pi = len(subdata) / len(data)
                newEntropy += pi * Ent(subdata, flag)
            infoGain = baseEntropy - newEntropy  # 中间信息增益
            print(Feature, " : ", infoGain, end=" ")
            if (infoGain > bestGain):  # 可以保序
                bestGain = infoGain
                bestFeature = Feature  # 返回信息增益最大的特征列名
        elif (flag == 2):
            uniquevalue = set(data[Feature])
            newEntropy = 0.0
            if (bestFeature == -1):
                bestFeature = Feature
            for value in uniquevalue:
                subdata = splitdataSet(data, Feature, value)
                pi = len(subdata) / len(data)
                newEntropy += pi * Ent(subdata, flag)
            print(Feature, " : ", newEntropy, end=" ")
            if (newEntropy < bestGini):
                bestGini = newEntropy
                bestFeature = Feature  # 返回信息增益最大的特征列名
        elif (flag == 3):
            newdata = deleteemptydataSet(data, Feature)
            baseEntropy = Ent(newdata, flag)
            uniquevalue = set(newdata[Feature])
            newEntropy = 0.0
            if (bestFeature == -1):
                bestFeature = Feature
            for value in uniquevalue:
                subdata = splitdataSet(newdata, Feature, value)
                pi = subdata[subdata.columns[-1]].sum(axis=0) / newdata[newdata.columns[-1]].sum(axis=0)  # 从列长度比变为列求和比
                #print(newdata)
                #print(subdata)
                newEntropy += pi * Ent(subdata, flag)  # 最后一列是权重
            infoGain = baseEntropy - newEntropy  # 中间信息增益
            infoGain = infoGain * newdata[newdata.columns[-1]].sum(axis=0)/ data[data.columns[-1]].sum(axis=0)
            # 比例p不能忘
            print(Feature, " : ", infoGain, end=" ")
            if (infoGain > bestGain):
                bestGain = infoGain
                bestFeature = Feature
    # print(bestFeature)

    node_num = node_num + 1
    print()
    return bestFeature


# 建立决策树  （DataFrame）（返回dict）
def createtree(data, flag):
    if (flag == 1 or flag == 2):
        labels = data.columns
        f = data[labels[-1]]
        if (len(f.values) == f.value_counts()[0]):  # 结束条件1：该分支f相同
            return f.values[0]
        if (len(labels) == 1):  # 结束条件2：所有属性循环完
            return f.value_counts().sort_values(ascending=False).index[0]  # 这里并不能直接返回f，可能不唯一
        bestFeature = splitbest(data, flag)
        myTree = {bestFeature: {}}  # 巧妙，创建嵌套字典树
        for value in set(data[bestFeature]):
            myTree[bestFeature][value] = createtree(splitdataSet(data, bestFeature, value), flag)  # 递归创建树
    elif (flag == 3):
        labels = data.columns
        f = data[labels[-2]]
        w = data[labels[-1]]
        labels = data.drop(['w'], axis=1).columns  # 不设置inplace，原数据不变
        if (len(f.values) == f.value_counts()[0]):
            return f.values[0]
        if (len(labels) == 1):
            return f.value_counts().sort_values(ascending=False).index[0]
        bestFeature = splitbest(data, flag)
        myTree = {bestFeature: {}}
        labels = {}
        sum_nonempty = 0
        names = data[bestFeature]  # 目标属性分布统计，不包括空值
        for i, j in names.value_counts().items():
            if (i != -1):
                labels[i] = j
                sum_nonempty += j
        for i in labels:
            subdata = splitdataSet(data, bestFeature, i)  # 待添加
            #print(subdata)
            for m in range(len(data)):  # 查找修改空值行
                if (data.iloc[[m], :][bestFeature].values[0] == -1):
                    temp = data.iloc[[m], :]  # 匹配到的这一行
                    temp.drop(bestFeature, axis=1, inplace=True)  # 易忘：删去目标属性列再添加
                    # newdata = deleteemptydataSet(data, bestFeature)
                    # pi = subdata[subdata.columns[-1]].sum(axis=0) / newdata[newdata.columns[-1]].sum(axis=0)
                    # temp['w'] = pi * temp['w']
                    subdata = subdata.append(temp, ignore_index=True)
                    subdata.loc[len(subdata) - 1, 'w'] = labels[i] * subdata.loc[len(subdata) - 1, 'w'] / sum_nonempty
            #print(subdata)
            myTree[bestFeature][i] = createtree(subdata, flag)
    return myTree





if __name__=="__main__":
    # 第二题：划分
    # data = pd.DataFrame({
    #     'X': [1, 1,0,0,0,0,1, 1], 'Y': [0,1,0,1, 1, 0, 0, 1],'Z': [1, 0, 0, 1,0,1,0,1 ], 'f': ['1','0','0','1','0','0','0','0']})

    # 第三题：剪枝
    # data = pd.DataFrame({
    #      '爱运动': ['是','否','是','是','否'], '爱学习': ['是','是','否','否','否'], '成绩高': ['是','是','否','否','是']})


    # 第四题：缺失
    data = pd.DataFrame({
        'X': [1, -1, 0, 0, -1, 0, 1, 1], 'Y': [0, 1, -1, 1, 1, 0, -1, 1], 'Z': [-1, 0, 0, 1, 0, -1, 0, 1],
        'f': ['1', '0', '0', '1', '0', '0', '0', '0']})
    data = data.copy(deep=True)
    data['w'] = 1


    # flag=1信息增益，flag=2基尼指数，flag=3缺失值
    print(createtree(data, 3))
    createPlot(createtree(data, 3))

