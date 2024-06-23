import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def fit_transform(data):
    processor = MinMaxScaler()
    return processor.fit_transform(data), processor


def split(X, t):
    X_train, X_val, t_train, t_val = \
        train_test_split(X, t, test_size=0.20,
                         random_state=42, shuffle=True)

    return X_train, X_val, t_train, t_val


def train_eval_process(model, X_train, X_val, t_train, t_val):
    model.fit(X_train, t_train)

    pred_train = model.predict(X_train)
    error_train = mean_squared_error(t_train, pred_train)

    pred_val = model.predict(X_val)
    error_val = mean_squared_error(t_val, pred_val)

    return error_train, error_val


 def transform_train_val(X_train, X_val, t_train, t_val):
    # proper transformation to data
    X_train, X_train_transformer = fit_transform(X_train)
    X_val = X_train_transformer.transform(X_val)

    t_train, t_train_transformer = fit_transform(t_train)
    t_val = t_train_transformer.transform(t_val)
    return X_train, X_val, t_train, t_val


def what_is_wrong():
    model = linear_model.LinearRegression()

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    t = diabetes.target.reshape(-1, 1)

    X, _ = fit_transform(X)
    t, _ = fit_transform(t)

    X_train, X_val, t_train, t_val = split(X, t)

    error_train, error_val = \
        train_eval_process(model, X_train, X_val, t_train, t_val)

    return error_val


def alter_data(X, t):
    np.random.seed(0)  # Fix random values order

    # For sake of demonstration, let's alter it
    # We want the val set to be different than train distribution
    X_train, X_val, t_train, t_val = split(X, t)

    # add random number for every value
    X_val += np.random.normal(0, 1, size=X_val.shape)

    # combine again
    X = np.vstack([X_train, X_val])
    t = np.vstack([t_train, t_val])

    return X, t


if __name__ == '__main__':
    model = linear_model.LinearRegression()

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    # For easy calculation make val 2D
    t = diabetes.target.reshape(-1, 1)
    X, t = alter_data(X, t) # to administrate perfpormance gab

    do_it_right = True      # TRY with true or false
    if do_it_right:
        # split, preprocess train and use its parameters for val/test
        X_train, X_val, t_train, t_val = split(X, t)

        X_train, X_val, t_train, t_val = \
            transform_train_val(X_train, X_val, t_train, t_val)

    else:
        # WRONG: preprocess the whole data, then split
        X, _ = fit_transform(X)
        t, _ = fit_transform(t)

        X_train, X_val, t_train, t_val = split(X, t)

    error_train, error_val = train_eval_process(model, X_train, X_val, t_train, t_val)

    print(error_val)
    '''
    0.053   for the correct setup
    0.051   for the wrong setup ... it is LOWER but WRONG
    '''
