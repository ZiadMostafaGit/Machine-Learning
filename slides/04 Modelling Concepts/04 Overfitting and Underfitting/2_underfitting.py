import matplotlib.pyplot as plt
import numpy as np

# x^2 curve

n = 200
x = np.linspace(-20, 20, n)
mu, sigma = 0, 0.5
noise = np.random.normal(mu, sigma, n)
y = (x + noise) ** 2

plt.scatter(x, y)

plt.title(f'Size vs Price')
plt.xlabel('x: size')
plt.ylabel('y: price')
plt.grid()

plt.show()

