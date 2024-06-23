import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


def visualize2D(x, y, is_scatter=True, tite='x vs y'):
    if is_scatter:
        plt.scatter(x, y)
    else:
        plt.plot(x, y)
    plt.title(tite)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def get_linear_data(n = 500, m = 3):
    np.random.seed(0)
    x = np.linspace(0, 10, n)
    y = x + np.random.normal(0, 0.2, n)
    #visualize2D(x, y)
    # Append n x m random matrix
    x = x.reshape((-1, 1))
    x = np.hstack([x, np.random.normal(0, 1, size=(n, m))])
    return x, y


def evalaute(x, t, model, name):
    model.fit(x, t)
    pred_t = model.predict(x)
    err = mean_squared_error(t, pred_t)
    w = ' '.join([f'{w:.3f}' for w in model.coef_])
    print(f'{name}: MSE {err:.3f} - intercept {model.intercept_:.2f}- Weights {w}')


if __name__ == '__main__':
    x, t = get_linear_data()


    evalaute(x, t, Lasso(alpha=7),      'lasso 7    ')
    evalaute(x, t, Lasso(alpha=5),      'lasso 5    ')
    evalaute(x, t, Lasso(alpha=3),      'lasso 3    ')
    evalaute(x, t, Lasso(alpha=1),      'lasso 1    ')
    evalaute(x, t, Lasso(alpha=0.1),    'lasso 0.1  ')
    evalaute(x, t, Lasso(alpha=0.01),   'lasso 0.01 ')
    evalaute(x, t, Lasso(alpha=0.001),  'lasso 0.001')
    print()
    evalaute(x, t, Ridge(alpha=7)       , 'Ridge 7    ')
    evalaute(x, t, Ridge(alpha=1)       , 'Ridge 1    ')
    evalaute(x, t, Ridge(alpha=0.1)     , 'Ridge 0.1  ')
    evalaute(x, t, Ridge(alpha=0.01)    , 'Ridge 0.01 ')
    evalaute(x, t, Ridge(alpha=0.001)   , 'Ridge 0.001')

    #evalaute(x, t, LinearRegression(), 'LinearRegression')


    evalaute(x, t, Lasso(alpha=0.01), 'lasso 0.01 ')
    evalaute(x, t, Ridge(alpha=1), 'Ridge 1    ')
    evalaute(x, t, LinearRegression(), 'LinearRegression')
    print()
    x /= 10 ** 5
    # Tiny features require big weights.
    # But weights are penalized, and regualrizer will set to 0
    evalaute(x, t, Lasso(alpha=0.01), 'lasso 0.01 ')
    evalaute(x, t, Ridge(alpha=1), 'Ridge 1    ')
    evalaute(x, t, LinearRegression(), 'LinearRegression')

