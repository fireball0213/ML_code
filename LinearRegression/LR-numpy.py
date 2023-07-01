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
    w=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
    b=np.mean(Y-np.dot(X,w))
    return w,b

#使用numpy实现岭回归（lasso回归）
def ridge_regression(X,Y,alpha=0.1):
    # X是n*d的矩阵，Y是n*1的矩阵
    w=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+alpha*np.eye(X.shape[1])),X.T),Y)
    b=np.mean(Y-np.dot(X,w))
    return w,b

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


