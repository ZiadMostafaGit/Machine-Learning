import matplotlib.pyplot as plt
import numpy as np


x = np.array([1, 1.3, 2, 2.5, 2.7, 3, 3.25, 4], dtype=np.float32) * 100
y = 3*x - 50        # generate data on line

print('size', x)
print('price', y)

plt.scatter(x, y)
#plt.plot(x, y)

plt.title(f'Size vs Price')
plt.xlabel('x: size (meter^2)')
plt.ylabel('y: price (thousands)')
plt.grid()

plt.show()

