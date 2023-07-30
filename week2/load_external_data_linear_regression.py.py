import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy as cp
import math


def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    return X, y


x, y = load_data()


# for i in range(len(x)):
#     x[i] = float(x[i])
#     y[i] = float(y[i])
# for i, j in zip(x, y):
#     print(f"{i}, {j}")

# # Set the title
# plt.title("Profits vs. Population per city")
# # Set the y-axis label
# plt.ylabel('Profit in $10,000')
# # Set the x-axis label
# plt.xlabel('Population of City in 10,000s')
# plt.show()


def cost_function(x, y, w, b):
    m = x.shape[0]

    cost = 0
    for i in range(m):
        f_x = w*x[i]+b
        cost = cost+(f_x-y[i]**2)
    total_cost = (1/(2*m)*cost)
    return total_cost


def gradient_function(x, y, w, b):

    m = x.shape[0]

    dr_w = 0
    dr_b = 0

    for i in range(m):
        f_x = w*x[i]+b
        temp_w = (f_x-y[i])*x[i]
        temp_b = f_x-y[i]
        dr_w += temp_w
        dr_b += temp_b

    dr_w = dr_w/m
    dr_b = dr_b/m
    return dr_w, dr_b


def gradient_descent(x, y, w_init, b_init, alpha, num_item, cost_function, gradient_function):

    w = cp.deepcopy(w_init)
    w = w_init
    b = b_init

    for i in range(num_item):
        dr_w, dr_b = gradient_function(x, y, w, b)
        w = w-alpha*dr_w
        b = b-alpha*dr_b

    return w, b


w_init = 0
b_init = 0
alpha = 0.01
itrations = 1500
final_w, final_b = gradient_descent(
    x, y, w_init, b_init, alpha, itrations, cost_function, gradient_function)


# jlist, plist

print(f"(x,b) found by gradint descent : ({final_w:8.4f},{final_b:8.4f})")


# 1.1664, -3.6303
m = x.shape[0]
prodiction = np.zeros(m)
for i in range(m):
    prodiction[i] = final_w*x[i]+final_b


plt.plot(x, prodiction, c="b")
plt.scatter(x, y, marker='x', c='r')
plt.title("profits VS population per city")
plt.ylabel("profit in $10,000")
plt.xlabel("population of city in 10,000s")

# num = float(input("enter the X input: "))
# resulte = 1.1664*num+-3.6303
# print(f"the prodicted value for your input is :{resulte}")
plt.show()
