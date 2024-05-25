import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.arange(0, 100, 0.001)

mean = 70
for sigma in [5, 8, 12]:    # varying sigma
    y = norm.pdf(x, mean, sigma)    # evaluate the x
    label = f'standard deviation = {sigma}'
    plt.plot(x, y, label = label)


mean = 30
for sigma in [2]:           # moving mean
    y = norm.pdf(x, mean, sigma)
    plt.plot(x, y)

plt.xlabel('Grade')
plt.ylabel('Density')
plt.legend()
plt.show()
