
def normal_equations_solution(X, y):
    # (X.T * X)^(-1) * X.T * y
    from numpy.linalg import inv

    XT = X.T
    # Both ways have the same complexity, BUT the 2nd is faster
    #return inv(XT.dot(X)).dot(XT).dot(y)   # SLOW - see the time complexity question
    return inv(XT.dot(X)).dot(XT.dot(y))

