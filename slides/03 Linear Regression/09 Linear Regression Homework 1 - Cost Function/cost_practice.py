import numpy as np


def f(X, t, weights):
    examples = X.shape[0]
    pred = np.dot(X, weights)
    error = pred - t
    print(pred)
    print(error)
    # cost = np.sum(error ** 2) / (2 * examples)
    cost = error.T.dot(error) / (2 * examples)  # dot prodcut is WAY faster
    return cost


def f_dervative(X, t, weights):
    examples = X.shape[0]
    pred = np.dot(X, weights)  # same as x @ weights
    error = pred - t
    # For the jth weight, we need for all examples to multiply xj with error and sum them
    # This is equivalent to the following matrix multiplication
    # Use a simple example and verify
    gradient = X.T @ error / examples  # Same also as x.T.dot(error) / examples

    return gradient

if __name__ == '__main__':
    # Input is 1D feature, e.g. the price
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t = 5 + X   # Output linear, no noise

    X = X.reshape((-1, 1))  # let's reshape in 2D
    X = np.hstack([np.ones((X.shape[0], 1)), X])    # add 1 for c

    print(X.shape)  # 5 x 2: for line mx+c

    weights = np.array([1.0, 1.0])  # starting params

    print(f(X, t, weights)) # cost: 8

    print(f_dervative(X, t, weights))  # dervative: [-4.   -1.92]
