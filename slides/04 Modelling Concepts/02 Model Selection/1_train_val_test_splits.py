import numpy as np
from sklearn.model_selection import train_test_split

X = []
t = []
for i in range(1, 11, 1):   # 10 examples
    X.append([i, i * 10, i * 20])
    t.append(i * 100)
X = np.array(X)
t = np.array(t)
'''
[[  1  10  20]
 [  2  20  40]
 [  3  30  60]
 [  4  40  80]
 [  5  50 100]
 [  6  60 120]
 [  7  70 140]
 [  8  80 160]
 [  9  90 180]]
'''

# 20% for testing and 80% for (train+val)
X_trainval, X_test, t_trainval, t_test = train_test_split(X, t, test_size=0.20,
                                                    random_state=42,
                                                    shuffle = True)

'''
[[  6  60 120]          # X_trainval: 8 examples
 [  1  10  20]
 [  8  80 160]
 [  3  30  60]
 [ 10 100 200]
 [  5  50 100]
 [  4  40  80]
 [  7  70 140]]

[[  9  90 180]          # X_test: 2 examples
 [  2  20  40]]
'''

# 20/80 = 25% for val and 75% - Now overall: 60 - 20 - 20
X_train, X_val, t_train, t_val = train_test_split(X_trainval, t_trainval, test_size=0.25,
                                                    random_state=42,
                                                    shuffle = False)

'''
[[  6  60 120]          # X_train: 6 examples
 [  1  10  20]
 [  8  80 160]
 [  3  30  60]
 [ 10 100 200]
 [  5  50 100]]         
  
 [[  4  40  80]         # X_val: 2 examples
 [  7  70 140]]
 '''
