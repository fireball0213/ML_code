from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from performance_measure import MSE

X, y = load_boston(return_X_y=True)
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)


#使用numpy实现线性回归
def linear_regression(X,Y):
    # X是n*d的矩阵，Y是n*1的矩阵
    #简化w=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
    w=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))

    b=np.mean(Y-np.dot(X,w))
    return w,b

#使用numpy实现岭回归（lasso回归）
def ridge_regression(X,Y,alpha=0.1):
    # X是n*d的矩阵，Y是n*1的矩阵
    w=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+alpha*np.eye(X.shape[1])),X.T),Y)
    b=np.mean(Y-np.dot(X,w))
    return w,b



class LinearRegression :
    def __init__(self):
        '''初始化线性回归模型'''
        self.theta = None
    def fit_normal(self,train_data,train_label):
        '''
        input:train_data(ndarray):训练样本
              train_label(ndarray):训练标签
        '''
        #********* Begin *********#
        X=train_data
        X = np.hstack((X, np.ones([X.shape[0], 1])))#列拼接，13维特征变14维
        y=train_label
        self.theta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
        self.b=np.mean(y-np.dot(X,self.theta))
        #********* End *********#
        return self.theta
    def predict(self,test_data):
        '''
        input:test_data(ndarray):测试样本
        '''
        #********* Begin *********#
        X=test_data
        X = np.hstack((X, np.ones([X.shape[0], 1])))#测试数据同样也需要拼接
        return np.dot(X,self.theta)
        #********* End *********#

if __name__=="__main__":
    #线性回归并计算MSE
    w,b=linear_regression(trainx,trainy)
    print("线性回归的参数w为：",w)
    print("线性回归的参数b为：",b)
    print("线性回归的MSE为：",MSE(testy,np.dot(testx,w)+b))

    #岭回归并计算MSE
    w,b=ridge_regression(trainx,trainy)
    print("岭回归的参数w为：",w)
    print("岭回归的参数b为：",b)
    print("岭回归的MSE为：",MSE(testy,np.dot(testx,w)+b))


