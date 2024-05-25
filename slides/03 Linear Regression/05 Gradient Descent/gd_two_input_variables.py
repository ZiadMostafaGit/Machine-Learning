import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

def gradient_descent(fderiv_dx, fderiv_dy, inital_x, inital_y, step_size = 0.001, precision = 0.00001, max_iter = 1000):
    cur_xy = np.array([inital_x, inital_y])
    last_xy = np.array([float('inf'), float('inf')])
    xy_list = [cur_xy]

    iter = 0
    while norm(cur_xy-last_xy) > precision and iter < max_iter:
        print(cur_xy)
        last_xy = cur_xy.copy()     # must copy

        gx = fderiv_dx(cur_xy[0], cur_xy[1])
        gy = fderiv_dy(cur_xy[0], cur_xy[1])
        gradient = np.array([gx, gy])
        cur_xy -= gradient * step_size   # move in opposite direction

        xy_list.append(cur_xy)
        iter += 1

    print(f'The minimum z exists at (x,y) = {cur_xy}')

    return xy_list


def trial1():
    def f(x, y):
        return 3 * (x + 2) ** 2 + (y - 1) ** 2       # 3(x + 2)² + (y - 1)²

    # https://calculator-derivative.com/partial-derivative-calculator
    def fderiv_dx(x, y):
        return 6 * (x + 2)

    def fderiv_dy(x, y):
        return 2 * (y - 1)

    func_name = 'Gradient Descent on 2x² - 4x y + y⁴ + 2'

    inital_x, inital_y = -5.0, 2.5
    xy_list = gradient_descent(fderiv_dx, fderiv_dy, inital_x, inital_y)
    # The minimum z exists at (x,y) = [-2.00730307  1.20259678]


def trial2():
    def f(x, y):
        return 2 * x ** 2 + 4 * x * y + y ** 4 + 2       # 2x² - 4x y + y⁴ + 2

    # https://calculator-derivative.com/partial-derivative-calculator
    def fderiv_dx(x, y):
        return 4 * (x-y)

    def fderiv_dy(x, y):
        return 4 * (y **3 - x)

    func_name = 'Gradient Descent on 2x² - 4x y + y⁴ + 2'

    inital_x, inital_y = 2.5, 1.9
    xy_list = gradient_descent(fderiv_dx, fderiv_dy, inital_x, inital_y)
    # The minimum z exists at (x,y) = [1.11270719 1.04406617]


# Optional: https://glowingpython.blogspot.com/2012/01/how-to-plot-two-variable-functions-with.html


#

if __name__ == '__main__':
    trial1()
    #trial2()

