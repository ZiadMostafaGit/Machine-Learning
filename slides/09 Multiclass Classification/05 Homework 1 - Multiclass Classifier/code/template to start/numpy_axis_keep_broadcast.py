import numpy as np
import os

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})


def logsoftmax1(x):
    x = x - x.max()
    return np.log(np.exp(x) / np.sum(np.exp(x)))

def logsoftmax2(x):
    return x - x.max()

if __name__ == '__main__':
    x = np.array([1, 2, 3, 10])
    # [-9.0014 -8.0014 -7.0014 -0.0014]
    logsoftmax1(x)
    # [-9       -8      -7      0]
    logsoftmax2(x)



    arr2d = np.array([[1, 2, 3, 4],
                     [1, 3, 7, 8],
                     [2, 6, 10, 5]], dtype=np.int32)


    print(np.max(arr2d))            # 10
    print(np.max(arr2d, axis=0))    # [ 2  6 10  8]
    print(np.max(arr2d, axis=1))    # [ 4  8 10]

    arr2d = np.array([[1, 2, 3, 4],
                     [1, 3, 7, 8],
                     [2, 6, 10, 5]], dtype=np.int32)

    print(np.max(arr2d, axis=0, keepdims=True))
    # [[ 2  6 10  8]]           (1, 4)

    print(np.max(arr2d, axis=1, keepdims=True))
    # [[ 4] [ 8]  [10]]         (3, 1)

    arr2d = np.array([[1, 2, 3, 4],
                     [1, 3, 7, 8],
                     [2, 6, 10, 5]], dtype=np.int32)

    print(arr2d + np.array([[10, 20, 30, 40]], dtype=np.int32))
    '''
    [[11 22 33 44]
     [11 23 37 48]
     [12 26 40 45]]
    '''
    print(arr2d + np.array([[100], [200], [300]], dtype=np.int32))
    '''
    [[101 102 103 104]
     [201 203 207 208]
     [302 306 310 305]]
    '''