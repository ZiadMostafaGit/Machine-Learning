
import numpy as np
from numpy.linalg import norm



def costFunction(x, t, weight):
    cost = 0
    # the formula is 1/2n sum from 1 to n (y(x,w)-y)squared
    for i in range(len(x)):
        error = weight[0]
        for j in range(1, len(x[0])):
            error += x[i][j] * weight[j]
        
        error = t[i] - error
        error = error ** 2
        cost += error
    cost = cost / len(x)
    cost /= 2
    return cost


#1/n sum from i to n the error for each x with all its weights -t*x
def caleDrivateve(x,t,weight):
    newWeights=np.zeros_like(weight)
    for i in range(len(weight)):
        theNewWeight=0
        for j in range(len(x)):
            error=weight[0]
            for k in range(1,len(x[0])):
                error+=x[j][k]*weight[k]

            error=error-t[j]
            theNewWeight+=error*x[j][i]

        theNewWeight/=len(x)
        newWeights[i]=theNewWeight

    return newWeights
            


def LinearRegressionGradientDescent(x,t,inital_start, step_size = 0.001, precision = 0.00001, max_iter = 1000):
    cur_start = np.array(inital_start)
    last_start = cur_start + 100 * precision    # something different
    # start_list = [cur_start]

    iter = 0
    while norm(cur_start - last_start) > precision and iter < max_iter:
        print(costFunction(x,t,cur_start),end=" ")
        print(cur_start)
        last_start = cur_start.copy()     # must copy

        gradient = caleDrivateve(x,t,cur_start)
        cur_start -= gradient * step_size   # move in opposite direction

        # start_list.append(cur_start)
        iter += 1

    return cur_start




X = np.array([0, 0.2, 0.4, 0.8, 1.0])
t = 5 + X  # Output linear, no noise
X = X.reshape((-1, 1))  # let's reshape in 2
X = np.hstack([np.ones((X.shape[0], 1)), X])
weight = np.array([1.0, 1.0])
res=LinearRegressionGradientDescent(X,t,weight)
print(res)
