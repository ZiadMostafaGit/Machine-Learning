from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data():
    X = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/X.npy')
    y = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/y.npy')

    # Normalize data
    X = X / 255.0

    return X, y


X, y = load_data()

# Split data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Encoder compresses data into 32 dimensions from 784 dimensions
autoencoder = MLPRegressor(hidden_layer_sizes=(32,), activation='relu',
                           solver='adam', max_iter=5000, random_state=42)


autoencoder.fit(X_train, X_train)
X_test_reconstructed = autoencoder.predict(X_test)
mse = mean_squared_error(X_test, X_test_reconstructed)
print(f"Mean Squared Error on test data: {mse:.2f}")

# Visualize original and reconstructed images
n = 5  # how many digits we will display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_test_reconstructed[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()