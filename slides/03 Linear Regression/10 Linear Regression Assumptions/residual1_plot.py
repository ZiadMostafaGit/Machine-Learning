

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

x = np.random.rand(500)
noise = np.random.normal(0, 0.2 , 500)
t = x + noise   # diagonal line
x, t = x.reshape(-1, 1), t.reshape(-1, 1)

pred_t = linear_model.LinearRegression().fit(x, t).predict(x)
residuals = t - pred_t
plt.scatter(pred_t, residuals)
plt.xlabel('x: Prediction')
plt.ylabel('y: Residual')
plt.show()

