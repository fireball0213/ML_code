# -*- coding: UTF-8 -*- #
"""
@filename:plot_tree_from_dict.py
@author:201300086
@time:2023-07-02
"""
# 字典树的可视化部分
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="square", color='g', fc='0.9')  # 结点形状,boxstyle文本框类型,fc注释框颜色的深度
leafNode = dict(boxstyle="circle", color='b', fc='0.9')  # 定义叶结点形状
arrow_args = dict(arrowstyle="<-", connectionstyle='arc3', color='red')  # 定义父节点指向子节点或叶子的箭头形状


def plot_node(node_txt, center_point, parent_point, node_style):
    '''
    绘制父子节点，节点间的箭头，并填充箭头中间上的文本
    :param node_txt:文本内容
    :param center_point:文本中心点
    :param parent_point:指向文本中心的点
    '''
    createPlot.ax1.annotate(node_txt,
                            xy=parent_point,
                            xycoords='axes fraction',
                            xytext=center_point,
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=node_style,
                            arrowprops=arrow_args)


def get_leafs_num(tree_dict):  # 获取叶节点的个数
    leafs_num = 0
    # 字典的第一个键，也就是树的第一个节点
    root = list(tree_dict.keys())[0]
    # 这个键所对应的值，即该节点的所有子树。
    child_tree_dict = tree_dict[root]
    for key in child_tree_dict.keys():
        # 检测子树是否字典型
        if type(child_tree_dict[key]).__name__ == 'dict':
            # 子树是字典型，则当前树的叶节点数加上此子树的叶节点数
            leafs_num += get_leafs_num(child_tree_dict[key])
        else:
            # 子树不是字典型，则当前树的叶节点数加1
            leafs_num += 1
    return leafs_num


def get_tree_max_depth(tree_dict):  # 求树的最深层数
    max_depth = 0
    # 树的根节点
    root = list(tree_dict.keys())[0]
    # 当前树的所有子树的字典
    child_tree_dict = tree_dict[root]

    for key in child_tree_dict.keys():
        # 树的当前分支的层数
        this_path_depth = 0
        # 检测子树是否字典型
        if type(child_tree_dict[key]).__name__ == 'dict':
            # 如果子树是字典型，则当前分支的层数需要加上子树的最深层数
            this_path_depth = 1 + get_tree_max_depth(child_tree_dict[key])
        else:
            # 如果子树不是字典型，则是叶节点，则当前分支的层数为1
            this_path_depth = 1
        if this_path_depth > max_depth:
            max_depth = this_path_depth
    return max_depth


def plot_mid_text(center_point, parent_point, txt_str):
    '''
    计算父节点和子节点的中间位置，并在父子节点间填充文本信息
    :param center_point:文本中心点
    :param parent_point:指向文本中心点的点
    '''

    x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    createPlot.ax1.text(x_mid, y_mid, txt_str)
    return


def plotTree(tree_dict, parent_point, node_txt):
    '''
    绘制树
    :param tree_dict:树
    :param parent_point:父节点位置
    :param node_txt:节点内容
    '''
    leafs_num = get_leafs_num(tree_dict)
    root = list(tree_dict.keys())[0]
    # plotTree.totalW表示树的深度
    center_point = (plotTree.xOff + (1.0 + float(leafs_num)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 填充node_txt内容
    plot_mid_text(center_point, parent_point, node_txt)
    # 绘制箭头上的内容
    plot_node(root, center_point, parent_point, decisionNode)
    # 子树
    child_tree_dict = tree_dict[root]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # 因从上往下画，所以需要依次递减y的坐标值，plotTree.totalD表示存储树的深度
    for key in child_tree_dict.keys():
        if type(child_tree_dict[key]).__name__ == 'dict':
            plotTree(child_tree_dict[key], center_point, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plot_node(child_tree_dict[key], (plotTree.xOff, plotTree.yOff), center_point, leafNode)
            plot_mid_text((plotTree.xOff, plotTree.yOff), center_point, str(key))
    # h绘制完所有子节点后，增加全局变量Y的偏移
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
    return


def createPlot(tree_dict):
    fig = plt.figure(1, facecolor='white')  # 设置绘图区域的背景色
    fig.clf()  # 清空绘图区域
    axprops = dict(xticks=[], yticks=[])  # 定义横纵坐标轴,注意不要设置xticks和yticks的值!!!
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 由全局变量createPlot.ax1定义一个绘图区，111表示一行一列的第一个，frameon表示边框,**axprops不显示刻度
    plotTree.totalW = float(get_leafs_num(tree_dict))
    plotTree.totalD = float(get_tree_max_depth(tree_dict))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(tree_dict, (0.5, 1.0), '')
    plt.show()

#使用决策树字典，来进行预测的函数
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]#获取树的第一个特征
    secondDict = inputTree[firstStr]#获取第一个特征的值
    featIndex = featLabels.index(firstStr)#获取第一个特征的索引
    for key in secondDict.keys():#遍历第一个特征的值
        if testVec[featIndex] == key:#如果测试数据的第一个特征的值等于树的第一个特征的值
            if type(secondDict[key]).__name__ == 'dict':#如果树的第二个特征是字典型
                classLabel = classify(secondDict[key],featLabels,testVec)#递归调用分类函数
            else: classLabel = secondDict[key]#如果树的第二个特征不是字典型，则将树的第二个特征的值赋给classLabel
    return classLabel#返回分类结果

if __name__ == "__main__":
    dic = {2: {1: 0, 3: 1, 4: {0: {4: 2, 5: {3: {1: 1, 2: 2}}, 6: {1: {2: 1, 3: 1}}, 7: 1}}, 5: {3: {1: {0:
                                                 {5: 2, 6: {1: {2: 2, 3: 1}}, 7: 2}}, 2: 2}}, 6: 2}}

    dic2={2: {1: 0, 3: 1, 4: {0: {4: 2, 5: {3: {1: 1, 2: 2}}, 6: {1: {2: 1, 3: 1}}, 7: 1}}, 5: {3: {1: {0:
                                                  {5: 2, 6: {1: 2}, 7: 2}}, 2: 2}}, 6: 2}}
    createPlot(dic)
    createPlot(dic2)

