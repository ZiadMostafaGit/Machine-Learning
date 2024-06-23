import numpy as np
from matplotlib import pyplot as plt
from math import pi

cx, cy = 0, 0   # center
rx, ry = 1, 3   # radius
size = 100

for p in range(1, 10):
    # See wiki. We can generate ellipse using (cos, sin) in range [0, 2PI]
    t = np.linspace(0, 2*pi, size)
    x = cx+rx*np.cos(t)
    y = cy+ry*np.sin(t) #+ np.random.normal(0, 0.2, size)    # + normal noise

    #plt.scatter(t, y)
    #plt.scatter(t, x)
    plt.scatter(x**p, y**p)        # transform to line or something else
    #plt.scatter(abs(x), abs(y))     # mirror 3/4 the ellipse to the positive quarter
    plt.title(f'x^{p}, y^{p}')
    plt.grid()
    plt.show()

