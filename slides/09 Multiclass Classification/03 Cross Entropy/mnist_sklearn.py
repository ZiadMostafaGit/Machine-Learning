import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/X.npy')
y = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/y.npy')
# We typically normalize image pixels to be in range [0-1]
X = X / 255.0

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=20, alpha=1e-3,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)  # NN

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy}")
