import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler


def load_diabetes_scaled():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X, t = diabetes.data, diabetes.target
    X = MinMaxScaler().fit_transform(X)
    return X, t


def cost_f(X, t, weights):
    examples = X.shape[0]
    pred = np.dot(X, weights)
    error = pred - t
    cost = error.T.dot(error) / (2 * examples)
    return cost


def f_dervative(X, t, weights):
    examples = X.shape[0]
    pred = np.dot(X, weights)
    error = pred - t
    gradient = X.T @ error / examples
    return gradient


def gradient_descent_linear_regression(X, t, step_size = 0.01, precision = 0.0001, max_iter = 1000):
    examples, features = X.shape
    iter = 0
    cur_weights = np.random.rand(features)         # random starting point
    last_weights = cur_weights + 100 * precision    # something different

    print(f'Initial Random Cost: {cost_f(X, t, cur_weights)}')

    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()           # must copy
        gradient = f_dervative(X, t, cur_weights)
        cur_weights -= gradient * step_size
        #print(cost_f(X, cur_weights))
        iter += 1

    print(f'Total Iterations {iter}')
    print(f'Optimal Cost: {cost_f(X, t, cur_weights)}')
    return cur_weights




if __name__ == '__main__':
    #np.random.seed(0)  # If you want to fix the results

    X, t = load_diabetes_scaled()
    optimal_weights = gradient_descent_linear_regression(X, t)

    '''
    For load_diabetes_scaled
    
    Initial Random Cost: 14157.095353441024
    Total Iterations 1000
    Optimal Cost: 1780.7107007153581
    
    Tip: explore slower step size such as 0.0001
    '''

