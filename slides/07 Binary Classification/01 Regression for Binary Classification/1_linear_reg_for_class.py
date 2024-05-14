# Using Linear Regression for Classification: Why Fails

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

threshold = 0.6

def case1_linear():
    X = np.array([4, 5, 6, 7, 8, 9]).reshape((-1, 1))
    y = np.array([0, 0, 0, 1, 1, 1])


    model = linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    classes = (y_pred >= threshold).astype(int)
    print('ground', y)
    print('predct', classes)
    print(f'{np.count_nonzero(classes == y)} out of {y.size}')

    # visualization assumes first 3 are class #1
    plt.plot(X[:3], y[:3], 'o', color='b', label="Cat")
    plt.plot(X[3:], y[3:], '^', color='r', label="Dog")

    plt.plot(X, y_pred, color="blue", linewidth=3)  # plot line
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot([0, 10], [threshold, threshold], color="red", linewidth=3)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


def case2_linear():
    X = np.array([4, 5, 6, 7, 8, 9, 50]).reshape((-1, 1))
    y = np.array([0, 0, 0, 1, 1, 1, 1])


    model = linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    classes = (y_pred >= threshold).astype(int)
    print('ground', y)
    print('predct', classes)
    print(f'{np.count_nonzero(classes == y)} out of {y.size}')

    plt.plot(X[:3], y[:3], 'o', color='b', label="Cat")
    plt.plot(X[3:], y[3:], '^', color='r', label="Dog")

    plt.scatter(X, y, color="black")
    plt.plot(X, y_pred, color="blue", linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot([0, 50], [threshold, threshold], color="red", linewidth=3)
    plt.legend(loc="upper left")
    plt.grid()

    plt.show()



if __name__ == '__main__':
    case1_linear()
    case2_linear()
