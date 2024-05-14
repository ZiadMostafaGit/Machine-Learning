import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_breast_cancer_scaled(add_intercept):
    data = load_breast_cancer()
    X, t = data.data, data.target_names[data.target]
    n_classes = data.target.size
    t = (t == 'malignant').astype(int)

    X = MinMaxScaler().fit_transform(X)     # wrong but for simplicity (do on all)

    if add_intercept:    # Add intercept after scaling
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.3,
                                                        shuffle=True, stratify=t,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_f(X, t, weights):
    m = X.shape[0]
    prop = sigmoid(np.dot(X, weights))
    positive = -t * np.log(prop)
    negative = - (1 - t) * np.log(1 - prop)
    cost = 1 / m * np.sum(positive + negative)
    return cost


def f_dervative(X, t, weights): # almost like gradient descent
    m = X.shape[0]
    prop = sigmoid(np.dot(X, weights))
    error = prop - t
    gradient = X.T @ error / m
    return gradient


def gradient_descent_linear_regression(X, t, step_size = 0.1, precision = 0.0001, max_iter = 7000):    # no changes. Different params
    examples, features = X.shape
    iter = 0
    cur_weights = np.random.rand(features)         # random starting point
    last_weights = cur_weights + 100 * precision    # something different

    print(f'Initial Random Cost: {cost_f(X, t, cur_weights)}')

    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()           # must copy
        gradient = f_dervative(X, t, cur_weights)
        cur_weights -= gradient * step_size
        #print(cost_f(X, cur_weights))
        iter += 1

    print(f'Total Iterations {iter}')
    print(f'Optimal Cost: {cost_f(X, t, cur_weights)}')
    return cur_weights


def accuracy(X, t, weights, threshold = 0.5):
    m = X.shape[0]
    prop = sigmoid(np.dot(X, weights))
    labels = (prop >= threshold).astype(int)
    correct = np.sum((t == labels))
    return correct / m * 100.0


if __name__ == '__main__':
    #np.random.seed(0)  # If you want to fix the results


    if True:
        add_intercept = True  # Try with and without for our model!
        X_train, X_test, y_train, y_test = load_breast_cancer_scaled(add_intercept)
        optimal_weights = gradient_descent_linear_regression(X_train, y_train)
        print(accuracy(X_test, y_test, optimal_weights))
    else:
        # By default intercept is added
        X_train, X_test, y_train, y_test = load_breast_cancer_scaled(False)
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_test_prop = model.predict_proba(X_test)[:, 1]

        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        print('Training accuracy: %.4f' % accuracy_train)
        print('Test accuracy:     %.4f' % accuracy_test)

        report_train = classification_report(y_train, y_pred_train)
        report_test = classification_report(y_test, y_pred_test)
        print('Training\n%s' % report_train)
        print('Testing\n%s' % report_test)
