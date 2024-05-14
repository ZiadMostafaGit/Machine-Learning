import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.001)
mean, sigma = 0, 1
y = norm.pdf(x, mean, sigma)

plt.plot(x, y)

xs = list(range(-3, 3, 1))   # generate range from -3 to 3
ys = norm.pdf(x, mean, sigma)

for (x, y) in zip(xs, ys):
    # Draw vertical line at this x
    plt.axvline(x = x, color = 'r')     # , ymax= 0.5

plt.xlabel('Z-score')
plt.ylabel('Probability')
plt.show()
