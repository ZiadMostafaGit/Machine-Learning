import numpy as np


# Gaussian Probability Density Function
def gaussian_pdf(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi * std ** 2))) * \
           np.exp(-((x - mean) ** 2) / (2 * std ** 2))


class GNB:  # Gauss Naive Bayes
    def train(self, X, y):
        self.classes = np.unique(y)  # [0, 1, 2]
        self.means, self.stds, self.priors = {}, {}, {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.stds[c] = np.std(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X) # frequency

    def predict(self, X):
        num_samples = X.shape[0]
        preds = np.zeros(num_samples)

        for i in range(num_samples):
            posteriors = {}

            for c in self.classes:
                # compute probability of each independent feature
                props = gaussian_pdf(X[i], self.means[c], self.stds[c])
                # multiply 5 props to get likelihood
                likelihood = np.prod(props)
                # compute posteriors, without constant Z
                posteriors[c] = likelihood * self.priors[c]

            # return the index of the max class value
            preds[i] = max(posteriors, key=lambda k: posteriors[k])

        return preds


def get_data():
    # Generate data that represents 3 classes
    # for each class, generate 100 examples each of 5 features
    # all data follows gaussian (mean, sgima) are provided
    X0 = np.random.normal(2, 1, (100, 5))
    X1 = np.random.normal(4, 1, (100, 5))
    X2 = np.random.normal(6, 1, (100, 5))

    # Combine into one dataset
    X = np.vstack([X0, X1, X2])     # 300 x 5
    y = np.array([0] * 100 + [1] * 100 + [2] * 100) # classes
    # y: 100 0, then 100 1, then 100 2 for ground truth

    # Create some test data
    X_test = np.array([[1.5, 2, 2.2, 1.9, 2.1],
                       [4.2, 3.8, 4.5, 3.9, 4.0],
                       [6.1, 5.9, 6.0, 6.2, 6.1]])

    return X, y, X_test


if __name__ == '__main__':


    X, y, X_test = get_data()
    gnb = GNB()
    gnb.train(X, y)
    print( gnb.predict(X_test) )
