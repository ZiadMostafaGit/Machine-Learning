# ChatGPT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def compute_wcss(data, centroids, labels):
	wcss = 0
	for i, point in enumerate(data):
		centroid = centroids[labels[i]]
		wcss += np.linalg.norm(point - centroid)
	return wcss / len(data)

# Generate isotropic Gaussian blobs for clustering
data, true_labels = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.0)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
predicted_labels = kmeans.labels_


wss = compute_wcss(data, kmeans.cluster_centers_, kmeans.predict(data))
print(wss)




# Visualize
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Original Data
ax[0].scatter(data[:, 0], data[:, 1], c=true_labels,
              s=50, cmap='viridis', marker='o', edgecolors='w', linewidth=0.5)
ax[0].set_title('Original Data')

# KMeans Clustering
ax[1].scatter(data[:, 0], data[:, 1], c=predicted_labels,
              s=50, cmap='viridis', marker='o', edgecolors='w', linewidth=0.5)

centers = kmeans.cluster_centers_
ax[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
ax[1].set_title('KMeans Clustering')

plt.tight_layout()
plt.show()
