import numpy as np

from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression



def get_data():
    def fit_transform(data):
        processor = MinMaxScaler()
        return processor.fit_transform(data), processor

    diabetes = datasets.load_diabetes()
    x = diabetes.data
    t = diabetes.target.reshape(-1, 1)

    # For simplicity, but this is data leakage
    x, _ = fit_transform(x)
    t, _ = fit_transform(t)

    return x, t


def evalaute(x, t, model, name):
    model.fit(x, t)
    pred_t = model.predict(x)
    err = mean_squared_error(t, pred_t)
    print(f'{name}: MSE {err:.3f}')
    print(model.intercept_)
    print(abs(model.coef_).mean())


if __name__ == '__main__':
    x, t = get_data()

    grid = {}
    grid['alpha'] =  np.array([0.1, 1, 0.01])
    grid['fit_intercept'] = np.array([False, True])

    kf = KFold(n_splits=4, random_state=35, shuffle=True)
    search = GridSearchCV(Ridge(), grid,
                          scoring='neg_mean_squared_error', cv=kf)

    search.fit(x, t)

    for key, value in search.cv_results_.items():
        print(key, value)

    print('Best Parameters:', search.best_params_)
    # Best Parameters: {'alpha': 1.0, 'fit_intercept': False}
    model = Ridge(**search.best_params_)
    evalaute(x, t, model, 'Ridge')
