import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

X = np.array([101, 102, 103, 301, 302, 305, 400, 501, 502])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
groups = np.array([10, 10, 10, 30, 30, 30, 40, 50, 50])

logo = LeaveOneGroupOut()

for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    print(X_test)
'''
[101 102 103]
[301 302 305]
[400]
[501 502]
'''