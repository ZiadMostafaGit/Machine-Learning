# conda install statsmodels
# pip install statsmodels


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


# sample Xs from standard normal distribution: 99.7 data in [-3, 3]
x_norm = stats.norm.rvs(size=500)		# rv = random variable
# compare data vs normal distribution
sm.qqplot(x_norm, dist=stats.norm, line='45')   # line='45' => y = x

plt.xlabel('x from unknown')
plt.ylabel('x from normal')
plt.show()

x_uniform1 = np.arange(0, 1, 0.01)          # sequential data [0, 1] incremented with 0.01 => uniform data
x_uniform2 = np.linspace(-3 ,3, 500)        # sequential data in range [-3, 3]
x_uniform3 = np.random.rand(500) * 6 - 3    # uniform data in range [-3, 3]

sm.qqplot(x_uniform1, dist=stats.norm, line='45')
plt.show()

sm.qqplot(x_uniform2, dist=stats.norm, line='45')
plt.show()


sm.qqplot(x_uniform1, dist=stats.uniform, line='45')
plt.show()


sm.qqplot(x_uniform2, dist=stats.uniform, line='45')
plt.show()
