import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(f_deriv, inital_x, step_size = 0.001, precision = 0.00001):
    cur_x = inital_x                     # initial start
    last_x = float('inf')
    x_list = [cur_x]             # let's maintain our x movements

    while abs(cur_x - last_x) > precision:
    #for iter in range(100):
        #print(cur_x)
        last_x = cur_x

        gradient = f_deriv(cur_x)
        cur_x -= gradient * step_size   # move in opposite direction

        x_list.append(cur_x)            # keep copy of what we visit

    print(f'The minimum y exists at x {cur_x}')
    #print(x_list)
    # 7.5, -7.09, -6.704, -6.34, -6.001, -5.68, -5.38 .... -0.66682, -0.66681
    return x_list


def visualize(f_func, range_start, range_end, x_list, plt_title):
    # Let's visualize points movements
    x = np.linspace(range_start, range_end, 50)
    y = f_func(x)

    plt.plot(x, y)
    plt.title(plt_title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()

    for idx, xp in enumerate(x_list[::3]):   # draw our movements
        yp = f_func(xp)
        color = 'ro' if idx%2 else 'bo'      # switch between rd and blue
        plt.plot(xp, yp, color)

    plt.show()
    ####################################################################


def trial1():
    def f(x):
        return 3 * x ** 2 + 4 * x + 7       # 3x² + 4x + 7


    def f_derivative(x):
        return 6 * x + 4                    # derivative of f(x)

    func_name = 'Gradient Descent on 3x^2 + 4x + 7'

    for inital_x in [-7.5, 5, -2/3]:
        title = f'{func_name}: starting from {inital_x}'
        x_list = gradient_descent(f_derivative, inital_x, step_size = 0.01)
        visualize(f, -10, 10, x_list, title)

    '''
    The minimum y exists at x -0.6668200405024292
    The minimum y exists at x -0.666513535786557
    The minimum y exists at x -0.6666666666666666
    '''

def trial2():
    def f(x):
        return x ** 4 - 6 * x ** 2 - x - 1  # x⁴ - 6x² - x -1

    def f_derivative(x):
        return 4 * x ** 3 - 12 * x - 1      # 4x³ − 12x − 1


    func_name = 'Gradient Descent on x⁴ - 6x² - x - 1'

    for inital_x in [-2.4, -0.15, 0.1, 2.39]:
        title = f'{func_name}: starting from {inital_x}'
        x_list = gradient_descent(f_derivative, inital_x, step_size = 0.001)
        visualize(f, -2.5, 2.5, x_list, title)

        '''
        The minimum y exists at x -1.6892154238200836
        The minimum y exists at x -1.6883429437023425
        The minimum y exists at x 1.7719305891118124
        The minimum y exists at x 1.7726747523515027
        '''

if __name__ == '__main__':
    trial1()
    #trial2()
