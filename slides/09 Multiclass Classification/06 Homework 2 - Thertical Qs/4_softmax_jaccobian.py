import numpy as np


def softmax(x):
    x = x - x.max()
    return np.exp(x) / np.sum(np.exp(x))


def softmax_grad_iterative(s):
    jacobian = np.zeros((len(s), len(s)))

    for i in range(len(jacobian)):
        for j in range(len(jacobian)):
            if i == j:
                jacobian[i][j] = s[i] * (1-s[i])
            else: 
                jacobian[i][j] = -s[i] * s[j]
    return jacobian



def softmax_grad_vectorized2(s):
    '''
    Let's pretend the matrix is just
    -y1y1   -y1y2    -y1y3
    -y2y1   -y2y2     -y2y3
    etc

    This is just the - the outer product of Y  ( Nx1 * 1xN)
    Now the only missing one is the diagonal needs an extra term Y

    So in total the answer is diagonal - outer(y, y)

    '''
    diagonal = np.diag(s)
    outer = np.outer(s, s)
    jacobian = diagonal - outer
    return jacobian


if __name__ == '__main__':

    s = softmax(np.array([5, 7, 8]))
    print(softmax_grad_iterative(s), '\n')
    '''
    [[ 0.03388568 -0.00911326 -0.02477242]
     [-0.00911326  0.19215805 -0.18304478]
     [-0.02477242 -0.18304478  0.2078172 ]] 
    '''

    print(softmax_grad_vectorized2(s), '\n')

