import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import load_house_data
from lab_utils_common import dlc
import copy
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']


x, y = load_house_data()

w_init = [0.323432, 18.33442244, -53.467229, -26.38101922]


def cost_function(x, y, w, b):

    m = x.shape[0]

    cost = 0.0
    for i in range(m):
        f_w_i = np.dot(x[i], w)+b
        cost = cost+(f_w_i-y[i])**2

    total_cost = cost / (2*m)

    return total_cost


def gradient_fucntion(x, y, w, b):

    m, n = x.shape
    dr_w = np.zeros(n)

    dr_b = 0

    for i in range(m):
        err = (np.dot(x[i], w)+b)-y[i]
        for j in range(n):
            dr_w[j] = dr_w[j]+err*x[i, j]
        dr_b = dr_b+err

    dr_w = dr_w/m
    dr_b = dr_b/m
    return dr_w, dr_b


def gradient_descent(x, y, w_init, b_init, alpha, num_item, gradient_function):

    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_item):
        dr_w, dr_b = gradient_fucntion(x, y, w, b)

        w = w-alpha*dr_w
        b = b-alpha*dr_b

        # if i < 10000:
        #     j.append(cost_functio(x, y, w, b))
        # if i % math.ceil(num_item/10) == 0:
        #     print(f"itartion {i:4d}:cost {j[-1]:8.2f}")
    return w, b


w_init = np.zeros_like(w_init)
b_init = 0
alpha = 1e-6
itration = 1000


# #

# mu = np.mean(x)
# segma = np.std(x)

# x_norm = (x-mu)/segma


# final_w, final_b = gradient_descent(
#     x, y, w_init, b_init, alpha, itration, gradient_fucntion)

# print(f"w,b found by gradient_descent: {final_w} {b_init:0.2f}")
# print(x_norm[1])
# new_w = np.array([134.06, -41.72, -41.88, -39.23])

# new_b = 0.00
# new_x = [2.46, -0.56, -0.57, -0.54]


# res = np.dot(new_w, new_x)+new_b
# print(res)
# print(y[1])
print(x)
print(y)
