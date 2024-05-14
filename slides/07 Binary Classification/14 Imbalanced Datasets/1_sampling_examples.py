from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

import numpy as np
from numpy import where
# pip install scikit-learn==1.0.2
# pip install imblearn  (imb for imbalanced) - neeed specific scikit version?


def project(X):
    if X.shape[1] > 2:
        # Project to 2 features
        # https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/
        tsne = TSNE(n_components=2, random_state=1)
        X = tsne.fit_transform(X)
    return X


def visualize_v2(X, y):
    X = project(X)

    counter = Counter(y)
    # scatter plot of examples by class label
    for label, freq in counter.items():
        row_ix = where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    plt.show()


def visualize(X, y):
    X = project(X)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.colorbar()
    plt.show()


def under_sample(X, y):
    counter = Counter(y)
    print(counter)  # Counter({1: 4923, 0: 77})

    factor, minority_size = 2, counter[0]   # 77
    rus = RandomUnderSampler(sampling_strategy={1: factor * minority_size}, random_state=42)
    X_us, y_us = rus.fit_resample(X, y)
    print(Counter(y_us))                    # Counter({1: 154, 0: 77})

    return X_us, y_us


def over_sample(X, y):
    counter = Counter(y)            # Counter({1: 4923, 0: 77})
    factor, majoirty_size = 1, counter[1]  # 4923
    new_sz = int(majoirty_size / factor)

    oversample = SMOTE(random_state=1, sampling_strategy={0: new_sz}, k_neighbors=3)
    #oversample = RandomOverSampler(random_state=1, sampling_strategy={0: new_sz})

    X_os, y_os = oversample.fit_resample(X, y)
    counter = Counter(y_os)                    # Counter({1: 4923, 0: 4923})
    print(counter)

    return X_os, y_os


if __name__ == '__main__':
    X, y = make_classification(n_samples=5000, n_features=5,
                               n_informative=2, n_redundant=3,
                               n_clusters_per_class=1, weights=[0.01], random_state=1)

    #visualize(X, y)

    #X, y = under_sample(X, y)
    #visualize(X, y)


    X, y = over_sample(X, y)
    #visualize(X, y)
