

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


x = np.random.normal(0.2, 0.09 , 10000)
x *= 100

plt.xlabel('Grade')
plt.ylabel('Number of Students')

n, bins, patches = plt.hist(x, bins=25, density=False)


plt.show()

