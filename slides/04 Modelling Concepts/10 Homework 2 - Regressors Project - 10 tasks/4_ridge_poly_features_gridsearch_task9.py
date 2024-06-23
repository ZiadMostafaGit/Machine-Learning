import argparse
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from data_helper import  load_data

KFLODS = 4
RANDOM_STATE = 17
np.random.seed(RANDOM_STATE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regressors Homework')

    parser.add_argument('--dataset', type=str, default='data2_200x30.csv')
    args = parser.parse_args()

    X, t, X_train, t_train, X_val, t_val = load_data(args.dataset)

    model = Ridge()
    pipeline = Pipeline(steps=[("scaler", MinMaxScaler()),
                               ('poly', PolynomialFeatures(degree=-1)),
                               ('model', model)])

    grid = {}   #
    grid['model__alpha'] =  np.array([1, 0.1, 10])
    grid['poly__degree'] = np.array([2, 1, 3])
    grid['poly__include_bias'] = np.array([False, False, False])
    grid['model__fit_intercept'] = np.array([False, True])

    kf = KFold(n_splits=KFLODS, random_state=RANDOM_STATE, shuffle=True)
    search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', cv=kf)
    search.fit(X, t)

    print('Best Parameters:', search.best_params_)
    model = search.best_estimator_

    pred_t = model.predict(X_val)
    err = mean_squared_error(t_val, pred_t, squared=False)
    print(f'RMSE {err:.3f}')
    print(model[-1].intercept_)
    print(abs(model[-1].coef_).mean())
