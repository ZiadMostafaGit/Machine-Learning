# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import StandardScaler
# from lab_utils_multi import load_house_data
# from lab_utils_common import dlc
# np.set_printoptions(precision=2)
# plt.style.use('./deeplearning.mplstyle')


# X_train, y_train = load_house_data()
# X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']


# scaler = StandardScaler()
# X_norm = scaler.fit_transform(X_train)

# sgdr = SGDRegressor(max_iter=1000)
# sgdr.fit(X_norm, y_train)
# print(sgdr)
# print(
#     f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import load_house_data
from lab_utils_common import dlc

np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=1000, alpha=1e-4)
sgdr.fit(X_norm, y_train)
print(sgdr.n_iter_)
print(X_norm[1])
print(y_train[1])


X = np.array(X_norm[1])
X = X.reshape(1, -1)
print(X)
res = sgdr.predict(X)
print(res)
