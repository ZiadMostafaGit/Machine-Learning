import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


def load_data():
    X = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/X.npy')
    y = np.load('/home/moustafa/0hdd/research/ndatasets/mnist/sample/y.npy')

    # Normalize data
    X = X / 255.0

    return X, y


def dtanh(y):   # tanh derivative
    return 1 - y ** 2


def softmax_batch(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_batch(y_true, y_pred):
    sample_cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-15), axis=1)
    return np.mean(sample_cross_entropy)


class NeuralNetworkMultiClassifier:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim1)
        self.b1 = np.zeros((1, hidden_dim1))

        self.W2 = np.random.randn(hidden_dim1, hidden_dim2)
        self.b2 = np.zeros((1, hidden_dim2))

        self.W3 = np.random.randn(hidden_dim2, output_dim)
        self.b3 = np.zeros((1, output_dim))

    def train(self, X_train, y_train, X_test, y_test, learning_rate = 1e-2, n_epochs = 20, batch_size = 32):
        def forward(X_batch):
            net1 = np.dot(X_batch, self.W1) + self.b1
            out1 = np.tanh(net1)
            net2 = np.dot(out1, self.W2) + self.b2
            out2 = np.tanh(net2)
            net3 = np.dot(out2, self.W3) + self.b3
            out3 = softmax_batch(net3)

            return out1, out2, out3

        def backward(y_batch, out1, out2, out3):
            dE_dnet3 = out3 - y_batch
            dE_dout2 = np.dot(dE_dnet3, self.W3.T)
            dE_dnet2 = dE_dout2 * dtanh(out2)
            dE_dout1 = np.dot(dE_dnet2, self.W2.T)
            dE_dnet1 = dE_dout1 * dtanh(out1)

            dW3 = np.dot(out2.T, dE_dnet3)
            db3 = np.sum(dE_dnet3, axis=0, keepdims=True)
            dW2 = np.dot(out1.T, dE_dnet2)
            db2 = np.sum(dE_dnet2, axis=0, keepdims=True)
            dW1 = np.dot(X_batch.T, dE_dnet1)
            db1 = np.sum(dE_dnet1, axis=0, keepdims=True)

            return dW1, db1, dW2, db2, dW3, db3

        def update(dW1, db1, dW2, db2, dW3, db3):
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3
            
        for epoch in range(n_epochs):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                out1, out2, out3 = forward(X_batch)
                dW1, db1, dW2, db2, dW3, db3 = backward(y_batch, out1, out2, out3)
                update(dW1, db1, dW2, db2, dW3, db3)
                loss = cross_entropy_batch(y_batch, out3)

            acc = self.test(forward, X_test, y_test)
            print(f"Epoch {epoch+1}, Last Loss: {loss}, Current Accuracy: {acc}")

    def test(self, forward, X_test, y_test):
        y_pred = forward(X_test)[-1]    # forward and get the predicition
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    X, y = load_data()
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    nn = NeuralNetworkMultiClassifier(X_train.shape[1], 20, 15, 10)

    nn.train(X_train, y_train, X_test, y_test)
