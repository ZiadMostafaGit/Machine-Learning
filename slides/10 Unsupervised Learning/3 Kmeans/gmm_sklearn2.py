# ChatGPT
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Create synthetic data
X, y = make_blobs(n_samples=400, centers=4, random_state=42, cluster_std=1.0)

# Plot the synthetic data
plt.scatter(X[:, 0], X[:, 1], c='blue', s=40, cmap='viridis')
plt.title("Original Data")
plt.show()

gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Get the coordinates of the cluster centers
probs = gmm.predict_proba(X)
# Output the parameters
print("Means:", gmm.means_)
print("Covariances:", gmm.covariances_)

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title("Data After GMM Clustering")
plt.show()


# Show the probabilities of the first 5 instances
print("Probabilities of the first 5 instances:")
print(probs[:5])