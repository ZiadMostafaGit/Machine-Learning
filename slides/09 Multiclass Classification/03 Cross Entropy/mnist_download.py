# download and save 5k sample for training

import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784")
X, y = mnist["data"].to_numpy(), mnist["target"].astype(int).to_numpy()

random_indices = np.random.choice(X.shape[0], size=5000, replace=True)
X = X[random_indices]
y = y[random_indices]
np.save('/home/moustafa/0hdd/research/ndatasets/mnist/sample/X.npy', X)
np.save('/home/moustafa/0hdd/research/ndatasets/mnist/sample/y.npy', y)

