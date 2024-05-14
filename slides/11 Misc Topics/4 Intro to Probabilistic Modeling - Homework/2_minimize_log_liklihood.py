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


def loglike():

    # generate 1000 examples from gaussian
    np.random.seed(0)

    m, s, n = 160, 15, 1000
    weights = np.random.normal(m, s, 1000)

    # Negative Log-Likelihood function for Gaussian
    def neg_log_likelihood(params):
        mean, std_dev = params
        n = len(weights)
        log_likelihood = -n/2 * np.log(2 * np.pi) - n/2 * np.log(std_dev**2) - np.sum((weights - mean)**2) / (2 * std_dev**2)
        return -log_likelihood


    initial_params = [np.mean(weights), np.std(weights)]

    # Minimize negative log likelihood
    result = minimize(neg_log_likelihood, initial_params, method='BFGS')
    estimated_mean, estimated_std_dev = result.x

    print(f"Estimated mean weight: {estimated_mean} lbs")
    print(f"Estimated standard deviation: {estimated_std_dev} lbs")


if __name__ == '__main__':
    simple()
    loglike()