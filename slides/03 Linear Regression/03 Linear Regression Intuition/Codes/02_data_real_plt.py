import matplotlib.pyplot as plt
import numpy as np

def scale(y):   # convert y to [0, 1] range. LATER
    mn, mx = np.min(y), np.max(y)
    y = (y - mn) / (mx - mn)
    return y

x = np.random.rand(500)         # 500 random value to visualize from uniform distribution

mu, sigma = 0, 0.3     # Mean and standard deviation
noise = np.random.normal(mu, sigma, 500)    # also see np.random.randn - n independent noise


if True:    # draw real line
    y = 3*x - 50
    y = scale(y)
    plt.plot(x, y, "r-", linewidth=4)    # red line

y = 3*x - 50 + noise
y = scale(y)
plt.scatter(x, y)

plt.title(f'Size vs Price')
plt.xlabel('x: size')
plt.ylabel('y: price')
plt.grid()

plt.show()

