import argparse
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from data_helper import  *


class GradientDescent:
    def __init__(self, step_size = 0.01, precision = 0.00001, max_iter = 1000000, lambd = 1.0):
        self.coef_ = None
        self.step_size = step_size
        self.precision = precision
        self.max_iter = max_iter
        self.lambd = lambd

    def fit(self, X, t):
        from numpy.linalg import norm

        examples, features = X.shape
        iter = 0
        cur_weights = np.random.rand(features, 1)
        
        state_history, cost_history = [], []
        last_weights = cur_weights + 100 * self.precision

        def f(weights):
            pred = np.dot(X, weights)
            error = pred - t
            cost = error.T.dot(error) / (2 * examples) 
            # lambda term
            cost += self.lambd / 2.0 * np.sum(weights * weights)  # weights * weights element-wise multiplication
            return cost

        def f_dervative(weights):
            pred = np.dot(X, weights)
            error = pred - t
            gradient = X.T @ error / examples
            
            # add lambda term
            gradient += self.lambd * weights   # don't divide by examples!
            return gradient

        while norm(cur_weights - last_weights) > self.precision and iter < self.max_iter:
            last_weights = cur_weights.copy()
            cost = f(cur_weights)
            gradient = f_dervative(cur_weights)

            state_history.append(cur_weights)
            cost_history.append(cost)

            cur_weights -= gradient * self.step_size
            iter += 1

        self.coef_ = cur_weights

    def predict(self, X):
        return np.dot(X, self.coef_)


def eval_model(model, X, t, keymsg, squared = False):
    t_pred = model.predict(X)
    error = mean_squared_error(t, t_pred, squared=squared)
    print(f'\t Error of {keymsg}: {error}')
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regressors Homework')
    parser.add_argument('--dataset', type=str, default='data2_200x30.csv')
    args = parser.parse_args()

    _, _, X_train, t_train, X_val, t_val = load_data(args.dataset)
    X_train, X_val = preprocess_data(X_train, X_val, poly_degree=1, preprocess_option=1)

    use_ours = True
    lambd = 1.0

    if use_ours:
        keymsg = 'Regularized Gradiet Descent'
        if True:
            # Allow bias term
            X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

        model = GradientDescent(step_size=0.001, precision=0.00001, max_iter=10000, lambd=lambd)
    else:
        keymsg = 'SKlearn Ridge'
        model = Ridge(lambd)    # uses normalied normal equations

    model.fit(X_train, t_train)
    eval_model(model, X_val, t_val, keymsg)
