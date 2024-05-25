import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def case3_logstic():
    # In practice, we scale input data
    if True:    # linearly separable
        X = np.array([[0, 2], [4, 1], [3, 10], [9, 1], [8, 7]])
        y = np.array([0, 0, 1, 1, 1])
    else:       # NOT linearly separable
        X = np.array([[0, 2], [3, 10], [9, 1], [8, 7]])
        y = np.array([0, 1, 1, 0])

    model = linear_model.LogisticRegression().fit(X, y)
    # SciKit uses default threshold = 0.5
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    print('ground', y)
    print('predct', y_pred)
    print(f'{np.count_nonzero(y_pred == y)} out of {y.size}')

    # Get parameters of mx+c equation / hyperplance
    b = model.intercept_[0]
    w1, w2 = model.coef_.T
    m, c = -w1 / w2, -b / w2    # won't work if w2 is zero

    plt.plot(X, m * X + c, color="blue", linewidth=3)   # decision boundary

    indices_cat = np.argwhere(y == 0).reshape(-1)
    X_cat = np.array([X[idx] for idx in indices_cat])
    plt.plot(X_cat[:, 0], X_cat[:, 1], 'o', color='b', label="Cat")

    indices_dog = np.argwhere(y == 1).reshape(-1)
    X_dog = np.array([X[idx] for idx in indices_dog])
    plt.plot(X_dog[:, 0], X_dog[:, 1], '^', color='r', label="Dog")


    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend(loc="best")
    plt.show()


'''
    # Unstable sikit feature to draw boundries
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py
    from sklearn.inspection import DecisionBoundaryDisplay
    _, ax = plt.subplots(figsize=(4, 3))
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        cmap=plt.cm.Paired,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
        xlabel="Sepal length",
        ylabel="Sepal width",
        eps=0.5,
    )
'''

if __name__ == '__main__':
    case3_logstic()
