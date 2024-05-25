import numpy as np

def cross_entropy(p, q):
    # Ensure the values are normalized
    #p = p / np.sum(p)
    #q = q / np.sum(q)

    # Adding a small constant for
    # numerical stability as log(0) is undefined
    ce =-np.sum(p * np.log(q + 1e-15))

    return ce


if False:
    p = np.array([0.1, 0.7, 0.2])  # True distribution
    q = np.array([0.1, 0.7, 0.2])  # Predicted distribution
    ce = cross_entropy(p, q)

cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.1, 0.7, 0.2]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.1, 0.6, 0.3]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.1, 0.5, 0.4]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.0, 0.5, 0.5]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.0, 0.4, 0.6]))
#cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.0, 0.3, 0.7]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.0, 0.2, 0.8]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.0, 0.1, 0.9]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.0, 0.01, 0.99]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.0, 0.0, 1]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([1.0, 0.0, 0]))
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0.5, 0.0, 0.5]))

