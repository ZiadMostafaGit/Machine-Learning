import numpy as np
from gradient_descent_vec import gradient_descent


if __name__ == '__main__':
    # https://calculator-derivative.com/partial-derivative-calculator

    def fun(x, y, z):
        return np.sin(x) + np.cos(y) + np.sin(z)

    def fderiv_dx(x):
        return np.cos(x)

    def fderiv_dy(y):
        return -np.sin(y)

    def fderiv_dz(z):
        return np.cos(z)

    def fderiv(state):
        derv = [fderiv_dx, fderiv_dy, fderiv_dz]
        gradients = []
        for dfunc, var in zip(derv, state):
            gradients.append(dfunc(var))
        return np.array(gradients)

    inital_x, inital_y, intial_z = 1, 2, 3.5
    state = np.array([inital_x, inital_y, intial_z])
    mn = gradient_descent(fderiv, state)
    mn_output = fun(mn[0], mn[1], mn[2])

    print(f'Initial start at {state} ends at point: {mn} with minimum value {mn_output}')
    # Initial start at [1.  2.  3.5] ends at point: [0.22457445 2.67782963 4.21311734] with minimum value -1.5496155799445872
    # Initial start at [ -7.5 -15.9 -12.1] ends at point: [ -7.72263523 -15.77876279 -13.06066696] with minimum value -2.463293552679835


