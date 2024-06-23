from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline


def fit_transform(data):
    processor = MinMaxScaler()
    return processor.fit_transform(data), processor


if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    t = diabetes.target.reshape(-1, 1)
    # The following is wrong (leakage), but for simplicity
    t, _ = fit_transform(t)

    degree = 3
    # pipeline is efficient and skip leakage of data split
    # but it is applied here on input X ONLY
    pipeline = make_pipeline(MinMaxScaler(),
                            PolynomialFeatures(degree),
                            LinearRegression(fit_intercept=True))

    kf = KFold(n_splits=4, random_state=35, shuffle=True)
    scores = cross_val_score(pipeline, X, t, cv = kf,
                             scoring = 'neg_mean_squared_error')
    scores *= -1    # change to mean_squared_error
    #print(scores)
    print(scores.mean(), scores.std())

    # There is also cross_val_predict
    # There is also cross_validate:
    #   - Allows multiple metrics / timings
    #   - Return the trained model (don't use it for production, train on most of data)
    # sklearn.compose.TransformedTargetRegressor: Transform the targets
