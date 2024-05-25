import matplotlib.pyplot as plt
import numpy as np


x1 = np.array([1, 5])
y1 = x1 + 2                 # compute their Ys

plt.plot(x1, y1, '-r')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.grid()

x2 = np.array([2.5, 2.5])
y2 = np.array([6.75, 4.5])
plt.plot(x2, y2, '-g', linewidth=4)

plt.text(2.5-0.5, 6.75, "target")
plt.text(2.5-0.1, 4.25, "prediction")
plt.text(1.75, 5.5, "ERROR")

x_pts = [1.5, 2, 3.5, 4, 3, 2.5]
y_pts = [3.5, 4, 5.5, 5.7, 5.25, 6.75]

for x, y in zip(x_pts, y_pts):
    plt.plot(x, y, 'bo')

plt.show()

