import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


np.random.seed(0)


def load_data():
    X = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/X.npy')
    y = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/y.npy')

    # Normalize data
    X = X / 255.0

    return X, y


def dtanh(y):   # tanh derivative
    return 1 - y ** 2


def softmax_batch(x):
    ..


def cross_entropy_batch(y_true, y_pred):
    ..


class NeuralNetworkMultiClassifier:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim1)
        self.b1 = np.zeros((1, hidden_dim1))

        self.W2 = np.random.randn(hidden_dim1, hidden_dim2)
        self.b2 = np.zeros((1, hidden_dim2))

        self.W3 = np.random.randn(hidden_dim2, output_dim)
        self.b3 = np.zeros((1, output_dim))

    def train(self, X_train, y_train, X_test, y_test, learning_rate = 1e-2, n_epochs = 20, batch_size = 32):
        ...
        # After each epoch, print the current accuracy
        """
		Epoch 1, Last Loss: 1.2066964081571263, Current Accuracy: 0.529
		Epoch 2, Last Loss: 0.9706821403353751, Current Accuracy: 0.594
		Epoch 3, Last Loss: 0.8100274752384711, Current Accuracy: 0.661
		Epoch 4, Last Loss: 0.7282755482354417, Current Accuracy: 0.695
		Epoch 5, Last Loss: 0.5844924162893406, Current Accuracy: 0.721
		Epoch 6, Last Loss: 0.5303509678170555, Current Accuracy: 0.725
		Epoch 7, Last Loss: 0.5058916701584597, Current Accuracy: 0.744
		Epoch 8, Last Loss: 0.4748391099488855, Current Accuracy: 0.743
		Epoch 9, Last Loss: 0.46299046147413275, Current Accuracy: 0.763
		Epoch 10, Last Loss: 0.4080112701695938, Current Accuracy: 0.76
		Epoch 11, Last Loss: 0.3405261454045503, Current Accuracy: 0.76
		Epoch 12, Last Loss: 0.3753233589001984, Current Accuracy: 0.765
		Epoch 13, Last Loss: 0.3636914961596305, Current Accuracy: 0.779
		Epoch 14, Last Loss: 0.3746632431709245, Current Accuracy: 0.781
		Epoch 15, Last Loss: 0.28652168039121356, Current Accuracy: 0.789
		Epoch 16, Last Loss: 0.36129966255969104, Current Accuracy: 0.788
		Epoch 17, Last Loss: 0.2846526447051336, Current Accuracy: 0.797
		Epoch 18, Last Loss: 0.2541237143423918, Current Accuracy: 0.791
		Epoch 19, Last Loss: 0.24874053035848015, Current Accuracy: 0.803
		Epoch 20, Last Loss: 0.23357926758846648, Current Accuracy: 0.795
        """

if __name__ == '__main__':
    X, y = load_data()
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    nn = NeuralNetworkMultiClassifier(X_train.shape[1], 20, 15, 10)

    nn.train(X_train, y_train, X_test, y_test)
