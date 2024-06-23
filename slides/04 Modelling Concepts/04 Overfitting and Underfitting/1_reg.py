import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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


def visualize_polynomial(model, degree, ptsX, ptsY, tite='x vs y', mn = 0, mx = 1):
    # Let's visualize the found polynomial
    # Add the training data + draw polynomial
    from sklearn.preprocessing import PolynomialFeatures
    # generate points and evalute, then draw
    x = np.linspace(mn, mx, 500)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    x_new = poly.fit_transform(x.reshape(-1, 1))
    t = model.predict(x_new)

    plt.scatter(ptsX, ptsY, c = 'red', linewidths=4)    # x and ground truth
    plt.plot(x, t)  # found polynomial
    plt.title(tite)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


def generate_data(generate_line= False, generate_full_sign= False):
    processor = MinMaxScaler()

    if generate_line:
        y = x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        return x, y

    if generate_full_sign:
        x = np.linspace(0, 60, 1000)
        y = np.sin(x)

        processor = MinMaxScaler()
        x = processor.fit_transform(x.reshape(-1, 1)).reshape(-1)
        return x, y

    # generate sample only from sin(x)
    x = np.array([0, 10, 20, 30, 40, 50, 60])
    y = np.sin(x)

    x = processor.fit_transform(x.reshape(-1, 1)).reshape(-1)
    mu, sigma = 0, 0.02
    noise = np.random.normal(mu, sigma, x.shape[0])
    y = processor.fit_transform(y.reshape(-1, 1)).reshape(-1) + noise
    return x, y



def learn_polynomial_sikit(X, t, degree=1, is_regularized = False):
    # degree =1: mx+c (line)
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures

    # we can convert X to [X^0, X^1, X^2, X^3]
    # include_bias: True: add the intercept (X^0)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_new = poly.fit_transform(X.reshape(-1, 1))  # (rows, degree + 1)
    # For degree = 3: 0.5 ==> array([1., 0.5, 0.25, 0.125])

    if is_regularized:
        model = Ridge()
        #model = Lasso()
    else:
        model = LinearRegression()

    # Normal code
    model.fit(X_new, t)
    pred_t = model.predict(X_new)
    err = mean_squared_error(t, pred_t)
    return err, model


if __name__ == '__main__':
    #X, t = generate_data(generate_line= True, generate_full_sign= False)
    #X, t = generate_data(generate_line=False, generate_full_sign=True)
    X, t = generate_data(generate_line=False, generate_full_sign=False)
    visualize2D(X, t)

    degrees, errors = [], []
    for degree in range(1, 8, 1):
        err, model = learn_polynomial_sikit(X, t, degree, is_regularized=False)
        avg_abs_weights_sum = abs(model.coef_).sum() / (degree+1)
        print(f'{degree}-th degree error: {err} - Average abs weight: {avg_abs_weights_sum}')
        #print(model.coef_)


        visualize_polynomial(model, degree, X, t, tite=f'Polynomial of degree {degree}', mn = 0, mx = 1)

        degrees.append(degree)
        errors.append(err)



    # Note: very high polynomial degrees are pointless
    # X^50: if X < 1, it vanish to zero. if X > 1, it explodes (big number)

    # Note, SKlearn like LinearRegression is based on normal equations
    # If we tried gradient descent, it will FAIL to find the minimum for the 7 points
    # The surface of the functions seems contain saddle points and we won't reach the minimm
