import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import load_house_data, run_gradient_descent
from lab_utils_multi import norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')


# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:, i], y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")
# # plt.show()
# fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:, i], y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")
# plt.show()

# set alpha to 9.9e-7

mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma


fig, ax = plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:, 0], X_train[:, 3])
ax[0].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
ax[1].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:, 0], X_norm[:, 3])
ax[2].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()
