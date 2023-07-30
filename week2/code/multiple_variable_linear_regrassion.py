import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy as cp
import math
np.set_printoptions(precision=2)

x = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])
w_init = [0.323432, 18.33442244, -53.467229, -26.38101922]


# def cost_function(x, y, w, b):
#     m = x.shape[0]
#     cost = 0.0

#     for i in range(m):
#         f_w_i = np.dot(x[i], w)+b
#         cost = cost+(f_w_i-y[i])**2

#     total_cost = cost/(2*m)


# def gradient_function(x, y, w, b):

#     m, n = x.shape
#     dr_w = np.zeros(n)
#     dr_b = 0
#     for i in range(m):
#         err = np.dot(x[i], w)-y[i]

#         for j in range(n):
#             dr_w = dr_w+err+x[i, j]

#         dr_b = dr_b+err
#     dr_w = dr_w/m
#     dr_b = dr_b/m
#     return dr_w, dr_b


# def gradient_descent(x, y, w_in, b_in, alpha, num_item, cost_function, gradient_function):

#     w = cp.deepcopy(w_in)
#     b = b_in
#     j = []

#     for i in range(num_item):
#         dr_w, dr_b = gradient_function(x, y, w, b)
#         w = w-alpha*dr_w
#         b = b-alpha*dr_b
#         if i < 10000:
#             j.append(cost_function(x, y, w, b))

#         if i % math.ceil(num_item/10) == 0:

#             print(f"itartion {i:4d}:cost {j[-1]:8.2f}")

#     return w, b, j


# w_initalize = np.zeros_like(w_init)
# b_init = 0
# alpha = 5.0e-7
# itration = 1000
# final_w, final_b, j_list = gradient_descent(
#     x, y, w_initalize, b_init, alpha, itration, cost_function, gradient_function)

# print(f"w,b found by gradient_descent: {final_w} {b_init:0.2f}")


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


def gradient_descent(x, y, w_init, b_init, alpha, num_item, cost_functio, gradient_function):

    w = cp.deepcopy(w_init)
    b = b_init
    j = []

    for i in range(num_item):
        dr_w, dr_b = gradient_fucntion(x, y, w, b)

        w = w-alpha*dr_w
        b = b-alpha*dr_b

        if i < 10000:
            j.append(cost_functio(x, y, w, b))
        if i % math.ceil(num_item/10) == 0:
            print(f"itartion {i:4d}:cost {j[-1]:8.2f}")
    return w, b, j


w_init = np.zeros_like(w_init)
b_init = 0
alpha = 5.0e-7
itration = 1000


final_w, final_b, j_list = gradient_descent(
    x, y, w_init, b_init, alpha, itration, cost_function, gradient_fucntion)

print(f"w,b found by gradient_descent: {final_w} {b_init:0.2f}")
