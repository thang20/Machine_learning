import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
plt.plot(X, y, 'ro') # r : red, o : hình tròn
plt.axis([140, 190, 45, 75]) # x từ 140-190, y từ 45 - 75
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
# plt.show()
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1) # axis=1 thêm vào ghép cột-cột, 0 : ghép hàng hàng
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print(w)

w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

plt.plot(X.T, y.T, 'ro')
plt.plot(x0, y0)
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
# plt.show()

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X, y)
print('result by scikit-learn  : ', linear_regression.coef_, linear_regression.intercept_)