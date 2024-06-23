import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures

def load_data(dataset_path):
    df = pd.read_csv(dataset_path)

    # replace missing values with median or mean
    df = df.fillna(df.median())

    data = df.to_numpy()
    # extract the target from the last column and the remaining data
    X = data[:, :-1]
    t = data[:, -1].reshape(-1, 1)

    # split to train part vs val part
    X_train = X[:100, :]
    t_train = t[:100, :]

    X_val = X[100:, :]
    t_val = t[100:, :]

    # don't scale data here - leave as a choice later
    return X, t, X_train, t_train, X_val, t_val


def get_preprocessor(preprocessor_option):
    if preprocessor_option == 0:
        return None

    if preprocessor_option == 1:
        return MinMaxScaler()

    return StandardScaler()


def fit_transform(data, preprocessor_option):
    preprocessor = get_preprocessor(preprocessor_option)

    if preprocessor is not None:
        data = preprocessor.fit_transform(data)

    return data, preprocessor


def monomials_poly_features(X, degree):
    '''
    For each feature xi, creates: xi^1, xi^2, xi^3...xi^degree without any cross features (xi * xj)
    '''
    assert degree > 0

    if degree == 1:
        return X

    examples = []
    # ToDo make it faster/pythonic? How?
    for example in X:
        example_features = []
        for feature in example:
            cur = 1
            feats = []
            for deg in range(degree):
                cur *= feature
                feats.append(cur)
            example_features.extend(feats)
        examples.append(np.array(example_features))

    return np.vstack(examples)


def preprocess_data(X_train, X_val=None, poly_degree=1, use_cross_features=True, preprocess_option=1):
    if poly_degree > 1:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)  # cross-features

        if use_cross_features:
            X_train = poly.fit_transform(X_train)
        else:
            X_train = monomials_poly_features(X_train, poly_degree)     # no cross features

        if X_val is not None:
            if use_cross_features:
                X_val = poly.fit_transform(X_val)
            else:
                X_val = monomials_poly_features(X_val, poly_degree)

    X_train, processor = fit_transform(X_train, preprocess_option)

    if X_val is not None and processor is not None:
        X_val = processor.transform(X_val)

    return X_train, X_val



/*

Is polynomial features first then scaling is the same as opposite?
No
Assume we have feature a in range mn, mx

Scale then mn/mx
a' = (a - mn) / (mx - nm)

one of poly features is just a'^2 =  (a - mn)^2 / (mx - nm)^2


If we poly first, 
then we have   (a^2 - mn^2) / (mx^2 - mn^2)

=====================

Which one should we apply first? Polynomial first.
Why? To that all features, whether original or polynomial, are on the same scale.
	Same scale: e.g. [0-1] range OR [0 mean, 1 std]

*/
