# ChatGPT

import numpy as np
from scipy.optimize import minimize

def simple():

    def objective_function(x):
        return (x - 3) ** 2

    initial_guess = [0.5]
    result = minimize(objective_function, initial_guess, method='BFGS')

    print("Optimized parameters:", result.x)
    print("Objective function value at minimum:", result.fun)



if __name__ == '__main__':
    simple()
