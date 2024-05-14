import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal


# cmap: color maps. We communicate information using colors
# https://matplotlib.org/stable/tutorials/colors/colormaps.html

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html

def visualize(xv, yv, zv):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')  # 1x1 plot (no other subplots)

    ax.plot_surface(xv, yv, zv, cmap=cm.jet)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def build_xy_pairs(n = 100):
    # generate (x, y) pairs centered around (0, 0) with radius r x 2r
    r = 2
    x = np.linspace(-r, r, n)
    y = np.linspace(-2*r, 2*r, n)

    # meshgrid is important. A bit hard to grasp. See code below for mesh
    xv, yv = np.meshgrid(x, y)  # xv is (nxn)
    xy = np.vstack((xv.flatten(), yv.flatten())).T  # (NxN, 2) shape - column based order

    return xv, yv, xy   # shapes: (N, N), (N, N), (NxN, 2)


def create_gaussian2d():
    n = 100

    # let's use simple gaussian mean and variance
    mu = np.array([0., 0.])
    covariance = np.array([[1., 0],
                           [0,  8.]])

    xv, yv, xy = build_xy_pairs(n)
    # create gaussian and evaluate the points
    normal_rv = multivariate_normal(mu, covariance)

    z = normal_rv.pdf(xy)   # (NxN) shape
    zv = z.reshape(n, n)    # (N, N) -                       skip order='F'

    return xv, yv, zv


def mesh_grid_order():
    x = np.array([10, 20, 30])
    y = np.array([3, 4, 5, 6])
    xv, yv = np.meshgrid(x, y)
    # for each y, for each x, sum y+x       [observe order]
    xy = (xv.flatten() + yv.flatten()).T

    print(xv)
    print('*****************')
    print(yv)
    print('*****************')
    print(xy)
    print('*****************')
    print(xy.reshape(4, 3))
    print('*****************')
    print(xy.reshape(3, 4, order='F'))


if __name__ == '__main__':
    #mesh_grid_order()
    xv, yv, zv = create_gaussian2d()    # each is 2D array
    visualize(xv, yv, zv)



#######################
# Optional
# Below code is an example how to write multivariate_gaussian pdf by yourself
def multivariate(xy, n = 100):
    mu = np.array([0., 0.])
    covariance = np.array([[1., 0],
                           [0,  8.]])

    # https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
        # Note: the visualization is wrong in this script (see order = F in this code)
    # An example of how to compute multivariate_gaussian density
    def multivariate_gaussian(xy, mu, covariance):
        """Return the multivariate Gaussian distribution on 2D array xy"""

        n = mu.shape[0]
        covariance_det = np.linalg.det(covariance)
        covariance_inv = np.linalg.inv(covariance)
        M = np.sqrt((2 * np.pi) ** n * covariance_det)

        # This einsum call calculates (x-mu)^T   .    covariance^-1     .   (x-mu)
        #   in a vectorized way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', xy - mu, covariance_inv, xy - mu)

        return np.exp(-fac / 2) / M

    z = multivariate_gaussian(xy, mu, covariance)
    z = z.reshape(n, n)
    return z
