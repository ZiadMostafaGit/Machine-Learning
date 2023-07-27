import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

plt.style.use('./deeplearning.mplstyle')

# the sits

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


# the cost function


def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_xb = w * x[i]+b
        cost = cost+(f_xb-y[i])**2
    total_cost = (1/(2*m)*cost)
    return total_cost


def gradint_function(x, y, w, b):

    m = x.shape[0]
    dr_w = 0
    dr_b = 0

    for i in range(m):
        f_x = w*x[i]+b
        temp_w = (f_x-y[i])*x[i]
        temp_b = (f_x-y[i])
        dr_w += temp_w
        dr_b += temp_b
    dr_w = dr_w / m
    dr_b = dr_b / m
    return dr_w, dr_b


def gradint_descent(x_train, y_train, w_in, b_in, num_item, alpha, cost_function, gradint_function):

    j = []
    w = copy.deepcopy(w_in)
    p = []
    w = w_in
    b = b_in

    for i in range(num_item):

        dr_w, dr_b = gradint_function(x_train, y_train, w, b)
        w = w - alpha*dr_w
        b = b-alpha*dr_b

        if i < 100000:

            j.append(cost_function(x_train, y_train, w, b))
            p.append([w, b])

        if i % math.ceil(num_item/10) == 0:
            print(f"Iteration {i:4}: Cost {j[-1]:0.2e} ",
                  f"dr_w: {dr_w: 0.3e}, dr_b: {dr_b: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, j, p


w_init = 0
b_init = 0
alpha = 1.0e-2
itrations = 10000

final_w, final_b, jlist, plist = gradint_descent(
    x_train, y_train, w_init, b_init, itrations, alpha, cost_function, gradint_function)


print(f"(x,b) found by gradint descent : ({final_w:8.4f},{final_b:8.4f})")

# plt_intuition(x_train, y_train)
# plt.close('all')
# fig, ax, dyn_items = plt_stationary(x_train, y_train)

# updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

# soup_bowl()
