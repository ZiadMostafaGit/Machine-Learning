import matplotlib.pyplot as plt
import numpy as np

##################################################################

def slope(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return dy / dx      # if dx is not zero!

point1 = (3, 0)
point2 = (-2, -10)
s = slope(point1, point2)   # 2 => y = 2*x + C

##################################################################

x = np.linspace(-5 ,5, 15)  # Return 15 evenly spaced numbers in range [5, 5]
y = 2*x - 6                 # compute their Ys


plt.plot(x, y, '-r', label= 'y = 2*x - 6')
plt.title(f'Graph for y= 2*x - 6')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.grid()

# Plot the 2 points
# bo and ro will draw a thick point: blue/red
plt.plot(point1[0], point1[1], 'bo')
plt.plot(*point2, 'ro')

plt.text(point1[0]-1, point1[1], f"{point1}")
plt.text(point2[0]-1.5, point2[1], f"{point2}")

plt.show()

