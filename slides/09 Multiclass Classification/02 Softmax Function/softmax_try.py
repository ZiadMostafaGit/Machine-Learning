import numpy as np





def sum_activate_v1(x):
    return x / x.sum()


sum_activate_v1(np.array([0, 2, 3]))     # [0.  0.4 0.6]
sum_activate_v1(np.array([5, 7, 8]))     # [0.25 0.35 0.4 ]



def sum_activate_v2(x):
    x = x - x.min()
    return x / x.sum()


sum_activate_v2(np.array([0, 2, 3]))     # [0.  0.4 0.6]
sum_activate_v2(np.array([5, 7, 8]))     # [0.  0.4 0.6]
sum_activate_v2(np.array([-2, 0, 1]))    # [0.  0.4 0.6]


def softmax(x):
    x = x - x.max()
    return np.exp(x) / np.sum(np.exp(x))


softmax(np.array([0, 2, 3]))     # [0.03511903 0.25949646 0.70538451]
softmax(np.array([5, 7, 8]))     # [0.03511903 0.25949646 0.70538451]



def soft_argmax(x):
    return np.sum(softmax(x) * range(x.size))


float_idx = soft_argmax(np.array([2, 4, 18, 3]))
print(int(round(float_idx)))    # 2

