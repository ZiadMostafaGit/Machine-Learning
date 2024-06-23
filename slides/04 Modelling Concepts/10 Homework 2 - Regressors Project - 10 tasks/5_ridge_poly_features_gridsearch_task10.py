import argparse
import numpy as np

from sklearn.base import BaseEstimator
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


'''
I managed to build this code without even looking to the official documentation
This is mainly due to accumulated experience. It was good anyway
as it turned out the available API/examples are basic and seems we need one more steps

The main idea was to create my class that contains the target model (e.g. linear)
    In OOP, this is called wrapping
Then I kept running and getting errors asking where is ****
Then adding its functionalities with my guess for what is behind it

I then had couple of errors that I needed to resolve
- I needed to return self even in set params. It is actually a common practice
- I needed to reset self.poly in the set function

As you see, this task is more about programming skills, OOP and APIs than machine learning.
You need such skills for the market as a good senior ML dev

Feel free to catch a mistake :)

Useful: 
    https://towardsdatascience.com/how-to-build-a-custom-estimator-for-scikit-learn-fddc0cb9e16e
    https://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
'''

class PolynomialRegression(BaseEstimator):
    def __init__(self, degree = 1, **kwargs):
        self.degree = degree
        # include_bias=False: let ridge add its own intecerpt based on parameters
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        if 'degree' in kwargs:
            del kwargs['degree']
        self.model = Ridge(**kwargs)

    def fit(self, X, y, **kwargs):
        X_new = self.poly.fit_transform(X)
        self.model.fit(X_new, y, **kwargs)
        return self

    def get_params(self, **kwargs):
        params = self.model.get_params(**kwargs)
        params['degree'] = self.degree
        return params

    def set_params(self, **kwargs):
        self.degree = kwargs['degree']
        # must recompute poly (as default is 1)
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=True)

        del kwargs['degree']
        self.model.set_params(**kwargs)
        return self # Must return self

    def predict(self, X):
        X_new = self.poly.fit_transform(X)
        p = self.model.predict(X_new)
        return p

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regressors Homework')

    parser.add_argument('--dataset', type=str, default='data2_200x30.csv')
    args = parser.parse_args()

    X, t, X_train, t_train, X_val, t_val = load_data(args.dataset)

    model = PolynomialRegression()  # our new class
    pipeline = Pipeline(steps=[("scaler", MinMaxScaler()), ('model', model)])

    grid = {}   #
    grid['model__alpha'] =  np.array([1, 0.1, 10])
    grid['model__degree'] = np.array([2, 1, 3])
    grid['model__fit_intercept'] = np.array([False, True])

    kf = KFold(n_splits=KFLODS, random_state=RANDOM_STATE, shuffle=True)
    search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', cv=kf)

    #search.fit(X_train, t_train)
    search.fit(X, t)                    # biased, but our actual test on a separate test set

    print('Best Parameters:', search.best_params_)
    model = search.best_estimator_  # best model pipeline of 2 things

    pred_t = model.predict(X_val)
    # This is biased performance (don't count on), as the parameters utilized validation part
    err = mean_squared_error(t_val, pred_t, squared=False)
    print(f'RMSE {err:.3f}')
    print(model[1].intercept_)
    print(abs(model[1].coef_).mean())
