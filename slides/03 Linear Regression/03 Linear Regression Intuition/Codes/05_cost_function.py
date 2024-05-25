import numpy as np



# some code sketching

# let's pretend answer is
M = [2, 3.5, 4, 10, 5]
C = [1, 6, 3]
# ground truth (X, Y)
X = [1, 2, 3, 4, 5, 6]
Y = [6, 5, 4, 3, 2, 1]

def compute_cost(m, c):
    cost = 0
    for (x, y_gt) in zip(X, Y):
        y_pd = m * x + c
        err = y_gt - y_pd
        squared_err = err ** 2
        cost += squared_err
    return cost / len(Y)

if __name__ == '__main__':
    best_cost =  float("inf")
    best_m, best_c = None, None
    for m in M:
        for c in C:
            this_cost = compute_cost(m, c)

            if best_cost > this_cost:
                best_cost = this_cost
                best_m, best_c = m, c

    print(best_m, best_c, best_cost)




