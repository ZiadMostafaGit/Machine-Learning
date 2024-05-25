# By ChatGPT :)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.0)
# X is (300, 2) and Y values [0, 1, 2] for 3 classes

# Create an instance of Logistic Regression wrapped in OneVsRestClassifier / OneVsOneClassifier
clf = OneVsRestClassifier(LogisticRegression())
#clf = OneVsOneClassifier(LogisticRegression())
clf.fit(X, y)

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Obtain confidence scores for each classifier
confidence_scores = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
predictions = np.argmax(confidence_scores, axis=1)

# Visualize decision boundaries for each binary classifier
for i, color in zip(range(len(clf.estimators_)), ['red', 'green', 'blue']):
    plt.contour(xx, yy, confidence_scores[:, i].reshape(xx.shape), levels=[0], colors=color)

# Plot the ambiguity region
plt.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.8, cmap='autumn', levels=2)

# Plot data points (X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k', cmap='autumn')

plt.title("One-vs-Rest with Ambiguity Region")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()