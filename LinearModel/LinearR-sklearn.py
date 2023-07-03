# -*- coding: UTF-8 -*- #
"""
@filename:LinearR-sklearn.py
@author:201300086
@time:2023-06-30
"""
from sklearn.linear_model import LinearRegression
x = [[4], [8], [12], [10], [16]]  # 横坐标
y = [3, 5, 7, 10, 15]  # 纵坐标
lr = LinearRegression().fit(x,y)  # 关键点
k = lr.coef_[0]
b = lr.intercept_
print(k, b)
print(lr.predict([[10]]))