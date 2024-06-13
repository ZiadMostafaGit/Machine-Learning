import numpy as np
from numpy.linalg import norm

# https://www.youtube.com/watch?v=QrzApibhohY
# https://www.youtube.com/watch?v=4Ct3Yujl1dk
# Use for debugging. Stop during actual trian/test
# Future note: Don't use it with deep learning drop out
def gradient_check(weights, f, f_dervative):
    gradients = f_dervative(weights)

    eps = 1e-4

    for idx in range(len(weights)):
        weights[idx] -= eps
        cost1 = f(weights)

        weights[idx] += 2 * eps
        cost2 = f(weights)

        weights[idx] -= eps  # restore

        gradient1 = gradients[idx]
        gradient2 = (cost2 - cost1) / (2 * eps)

        if not np.isclose(gradient1, gradient2, atol=0.001):
            print(f'{gradient1} vs {gradient2}')
            return False

    return True


def gradient_descent_linear_regression(X, t, step_size = 0.001, precision = 0.00001, max_iter = 1000):
    examples, features = X.shape
    iter = 0
    cur_weights = np.random.rand(features)         # random starting point
    #cur_weights = np.ones(features, dtype=np.float32)
    #cur_weights = np.array([0, 0.6338432,  0.20894728, 0.00150253])

    state_history, cost_history = [], []
    last_weights = cur_weights + 100 * precision    # something different

    def f(weights):
        pred = np.dot(X, weights)
        error = pred - t
        #cost = np.sum(error ** 2) / (2 * examples)
        cost = error.T.dot(error) / (2 * examples)  # dot prodcut is WAY faster
        return cost

    def f_dervative(weights):
        pred = np.dot(X, cur_weights)       # same as x @ cur_weights
        error = pred - t
        # For the jth weight, we need for all examples to multiply xj with error and sum them
        # This is equivalent to the following matrix multiplication
        # Use a simple example and verify
        gradient = X.T @ error / examples   # Same also as x.T.dot(error) / examples

        return gradient

    #assert gradient_check(cur_weights, f, f_dervative)

    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()           # must copy
        cost = f(cur_weights)
        gradient = f_dervative(cur_weights)

        #print(f'weights: {cur_weights}\n\tcost: {cost} - gradient: {gradient}')

        state_history.append(cur_weights)
        cost_history.append(cost)
        #print(f'state {state_history[-1]} has \n\tcost {cost_history[-1]} - gradient {gradient}')

        cur_weights -= gradient * step_size   # move in opposite direction
        iter += 1

    print(f'Number of iterations ended at {iter} - with cost {cost} - optimal weights {cur_weights}')
    return cur_weights, cost_history





