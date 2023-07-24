import numpy as np

import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use('./deeplearning.mplstyle')
#the sits
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730])

# f_xb = []

# the cost function
def cost_function(x, y, w, b):
    m = x.shape[0]
    cost_res = 0
    for i in range(m):
        f_xb[i] = w * x[i]+b
        cost = (f_xb[i]-y[i])**2
        cost_res += cost
    total_cost = (1/(2*m)*cost_res)
    return total_cost


plt_intuition(x_train, y_train)
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)

updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()
