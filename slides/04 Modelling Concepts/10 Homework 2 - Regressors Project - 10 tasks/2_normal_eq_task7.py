import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from data_helper import  load_data



class NormalEquations:
    def __init__(self, alpha = 1):
        self.alpha = alpha

    def get_error(self, x, t):
        examples, features = x.shape
        pred = np.dot(x, self.coef_)
        error = pred - t
        cost = error.T.dot(error) / (2 * examples)  # dot prodcut is WAY faster
        return cost

    def fit(self, X, y):
        # (X.T * X)^(-1) * X.T * y
        from numpy.linalg import inv

        examples, features = X.shape
        XT = X.T
        m = XT.dot(X) + np.identity(features) * self.alpha
        self.coef = inv(m).dot(XT.dot(y))
        self.intercept = 0
        self.err = self.get_error(X, y)

        return self.err, self.coef_

    def predict(self, X):
        return np.dot(X, self.coef_)

    @property
    def coef_(self):
        return self.coef

    @property
    def intercept_(self):
        return self.intercept


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regressors Homework')

    parser.add_argument('--dataset', type=str, default='data2_200x30.csv')
    args = parser.parse_args()

    _, _, X_train, t_train, X_val, t_val = load_data(args.dataset)

    for alpha in [0, 0.1, 1, 10, 100, 1000]:
        model = NormalEquations(alpha=alpha)
        #model = Ridge(alpha=alpha)

        model.fit(X_train, t_train)

        t_pred = model.predict(X_val)
        error = mean_squared_error(t_val, t_pred)

        #print(model.coef_)
        print(error)





