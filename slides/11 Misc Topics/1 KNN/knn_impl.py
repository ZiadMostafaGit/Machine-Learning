import numpy as np
from collections import Counter
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error


class KNN:
    def __init__(self, k=3, task='classification'):
        self.k, self.task = k, task

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [np.sum((x - x_train) ** 2) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]

        k_nearest_outputs = [self.y_train[i] for i in k_indices]
        if self.task == 'classification':
            most_common = Counter(k_nearest_outputs).most_common(1)
            return most_common[0][0]
        elif self.task == 'regression':
            return np.mean(k_nearest_outputs)

    def is_anomaly(self, x):
        threshold = 1.0
        distances = [np.sum((x - x_train) ** 2) for x_train in self.X_train]
        kth_distance = np.sort(distances)[self.k]
        return kth_distance > threshold * threshold


# Generate and test on classification dataset
# Our impl is way slower than sklearn impl
#X_class, y_class = make_classification(n_samples=10000, n_features=10, random_state=42)

X_class, y_class = make_classification(n_samples=500, n_features=2, n_redundant=0, n_clusters_per_class=1,
                                       random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2,
                                                                            random_state=42)

knn_class = KNN(k=3, task='classification')
knn_class.fit(X_train_class, y_train_class)
y_pred_class = knn_class.predict(X_test_class)

accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# Generate and test on regression dataset
X_regress, y_regress = make_regression(n_samples=500, n_features=2, noise=0.1, random_state=42)
X_train_regress, X_test_regress, y_train_regress, y_test_regress = train_test_split(X_regress, y_regress, test_size=0.2,
                                                                                    random_state=42)

knn_regress = KNN(k=3, task='regression')
knn_regress.fit(X_train_regress, y_train_regress)
y_pred_regress = knn_regress.predict(X_test_regress)

mse = mean_squared_error(y_test_regress, y_pred_regress)
print(f"Regression MSE: {mse:.2f}")
