import numpy as np
from sklearn.model_selection import KFold

X = []
t = []
for i in range(1, 10, 1):
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

kf = KFold(n_splits=3, random_state=None, shuffle=False)
#print(kf.get_n_splits(X))   # 3

for train_index, val_index in kf.split(X):
    print(f'Validation index: {val_index} - Training index{train_index}')
    # Extract data
    X_train, X_val = X[train_index], X[val_index]
    t_train, t_val = t[train_index], t[val_index]
'''
Validation index: [0 1 2] - Training index[3 4 5 6 7 8]
Validation index: [3 4 5] - Training index[0 1 2 6 7 8]
Validation index: [6 7 8] - Training index[0 1 2 3 4 5]
'''


# The first n_samples % n_splits folds have size n_samples // n_splits + 1,
# other folds have size n_samples // n_splits
kf = KFold(n_splits=4, random_state=None, shuffle=False)

for train_index, val_index in kf.split(X):
    #print(f'Validation index: {val_index} - Training index{train_index}')
    # Extract data
    X_train, X_val = X[train_index], X[val_index]
    t_train, t_val = t[train_index], t[val_index]
'''
Validation index: [0 1 2] - Training index[3 4 5 6 7 8]
Validation index: [3 4] - Training index[0 1 2 5 6 7 8]
Validation index: [5 6] - Training index[0 1 2 3 4 7 8]
Validation index: [7 8] - Training index[0 1 2 3 4 5 6]

We have 9 samples to divide on 4 groups
9/2 = 4
So 4 groups
but 2 * 4 = 9. Then the first group takes the extra sample
'''

kf = KFold(n_splits=4, random_state=None, shuffle=True)

for train_index, val_index in kf.split(X):
    print(f'Validation index: {val_index} - Training index{train_index}')
    # Extract data
    X_train, X_val = X[train_index], X[val_index]
    t_train, t_val = t[train_index], t[val_index]

'''
Validation index: [1 3 6] - Training index[0 2 4 5 7 8]
Validation index: [0 2] - Training index[1 3 4 5 6 7 8]
Validation index: [5 8] - Training index[0 1 2 3 4 6 7]
Validation index: [4 7] - Training index[0 1 2 3 5 6 8]

With each run, you get different ordering
you can fix the ordering by passing value to random_state
e.g. random_state = 1
We use this for reproducible results (others can generate the SAME results)
'''