import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def generate_data(n=200):
    processor = MinMaxScaler()

    x = np.random.uniform(-20, 20, n)
    # y = 5 + x + 4 * x ** 2 + x ** 3
    y = 5 + x + 4 * x ** 2 + 5 * x ** 3 - 8 * x ** 4

    #coff = [5, 1, 4, 5, -8] # cofficients
    #y = polyval(x, coff)    # Another way if we know the coefficients

    # scale x after generating y
    x = processor.fit_transform(x.reshape(-1, 1)).reshape(-1)

    mu, sigma = 0, 0.02
    noise = np.random.normal(mu, sigma, n)
    # easier add noise after transformation, as we know the data range
    y = processor.fit_transform(y.reshape(-1, 1)).reshape(-1) + noise

    return x, y


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


def visualize_polynomial(model, degree):
    # Let's visualize the found polynomial
    from sklearn.preprocessing import PolynomialFeatures
    # generate points and evalaute, then draw
    n = 500
    x = np.random.uniform(-50, 50, n)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    x_new = poly.fit_transform(x.reshape(-1, 1))
    t = model.predict(x_new)
    visualize2D(x, t)


def learn_polynomial(X, t, degree=1):  # deg =1: mx+c (line)
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures

    # we can just convert X to [X^0, X^1, X^2, X^3]
    # include_bias: True: add the intercept (X^0)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_new = poly.fit_transform(X.reshape(-1, 1))  # (rows, degree + 1)
    # For degree = 3: 0.5 ==> array([1., 0.5, 0.25, 0.125])

    # Normal code
    model = LinearRegression()
    model.fit(X_new, t)
    pred_t = model.predict(X_new)
    err = mean_squared_error(t, pred_t) / 2  # we can remove /2
    return err, model


def try_polynomials():
    X, t = generate_data()
    visualize2D(X, t)

    degrees, errors = [], []
    for degree in range(1, 7):
        err, model = learn_polynomial(X, t, degree)
        print(f'Degree {degree} has error {err} {abs(model.coef_).sum()} - {model.coef_}')

        visualize_polynomial(model, degree)

        degrees.append(degree)
        errors.append(err)

    visualize2D(degrees, errors, is_scatter=False, tite='Degree vs Error')

    '''
        y = 5 + x + 4 * x ** 2 + x ** 3                         : degree 3
        Degree 1 has error 0.003505199776382123
        Degree 2 has error 0.0031823079736543664
        Degree 3 has error 0.0001834989378579264                ***
        Degree 4 has error 0.0001827860336817794
        Degree 5 has error 0.00018260965152558484
        Degree 6 has error 0.00018258931984090155

        y = 5 + x + 4 * x ** 2 + 5 * x ** 3 - 8 * x ** 4        : degree 4
        Degree 1 has error 0.035259515944108484
        Degree 2 has error 0.002934140176136343
        Degree 3 has error 0.0029258075364497304
        Degree 4 has error 0.00023028330775762425               ***
        Degree 5 has error 0.00023010215115659195
        Degree 6 has error 0.0002238670873776721

    '''


def utility_draw_plane():  # separate for fun
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = np.meshgrid(range(10), range(10))
    # 𝑎𝑥 + 𝑏𝑦 + 𝑐𝑧 + 𝑑 = 0
    # Given x, y:    z = -(𝑎𝑥 + 𝑏𝑦 + d) / c
    zz = -(xx + yy + 9.0) / 2.0  # d = 9, c = 2
    # Also z = -a/c x -b/c y -d/c

    # plot the plane
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    plt.show()


def visualize_nonlinear_data_vs_linear_plane():
    # Plane equation: z = 2 * x + 4 * y + 5

    # Find z values for the data points (bell curve)
    x = np.linspace(-15, 15, 100)
    y = x ** 2  # polynomial feature - depend on x
    z = 2 * x + 4 * y + 5

    visualize2D(x, z)
    # draw curve scattered points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    # Enumerate region for plane visualization
    x2 = np.linspace(-25, 25, 300)
    y2 = np.linspace(0, 200, 300)  # don't have to be on polynomial (y2 != x2^2)
    xx, yy = np.meshgrid(x2, y2)
    # Compute the plane's z value
    zz = 2 * xx + 4 * yy + 5

    # draw plane in 3D
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    plt.show()


if __name__ == '__main__':
    # utility_draw_plane()
    # visualize_nonlinear_data_vs_linear_plane()
    try_polynomials()


