import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate 1D data from four Gaussians
np.random.seed(42)
data1 = np.random.normal(-10, 3, 200)
data2 = np.random.normal(0, 2, 300)
data3 = np.random.normal(8, 1, 200)
data4 = np.random.normal(18, 2, 300)

# Combine the data to form a single dataset
data = np.concatenate([data1, data2, data3, data4]).reshape(-1, 1)  # (1000, 1)

if False:
    # Plot the original data
    plt.figure()
    plt.hist(data, bins=50, density=True, alpha=0.6, color='b')
    plt.title('Original Data')
    plt.show()

# Generate data to visualize the GMM. Important to reshape
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)


# Fit a Gaussian Mixture Model
n_components = 4
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(data)

# score_samples returns the log not the actual probability
# convert back the log to probability
log_likelihood = gmm.score_samples(x)
gmm_y = np.exp(log_likelihood)

probs = gmm.predict_proba(data)     # density for each sample: (1000, 4):
labels = gmm.predict(data)          # highest density

gmm.means_, gmm.covariances_, gmm.weights_

# Plot the original data and the GMM components
plt.figure()
plt.hist(data, bins=50, density=True, alpha=0.6, color='b')
plt.plot(x, gmm_y, '-r')
plt.title(f'GMM Fitted to Data with n_components={n_components}')
plt.show()





def our_score_samples(gmm, x):
    def gaussian_density(x, mu, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * \
               np.exp(-0.5 * ((x - mu) / std) ** 2)

    sum_likelihood = np.zeros_like(x)
    zp = zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel())
    for mu, variance, pi in zp:
        sum_likelihood += pi * gaussian_density(x, mu, std=np.sqrt(variance))

    return np.log(sum_likelihood).reshape(-1)

log_likelihood_our = our_score_samples(gmm, x)

