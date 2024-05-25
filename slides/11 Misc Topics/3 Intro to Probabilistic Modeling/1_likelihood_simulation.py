import numpy as np
import gif      # pip install gif
import matplotlib.pyplot as plt

# Generate synthetic weight data for college students
np.random.seed(0)
num_samples = 1000
true_mean = 80
true_sigma = 5
data = np.random.normal(true_mean, true_sigma, num_samples)


def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def likelihood_gaussian(data, mu, sigma):
    return np.prod(gaussian(data, mu, sigma))


def log_likelihood_gaussian(data, mu, sigma):
    return np.sum(np.log(gaussian(data, mu, sigma)))



# Settings for Gaussian curve visualization
x = np.linspace(min(data)-20, max(data)+20, 1000)
fixed_sigma = 5
mean_range = np.linspace(60, 100, 10)  # Varying the mean from 79.5 to 80.5

# Plot Gaussian curves with incremental steps of mean
letters = 'ABCDEFGHIJKLMOPQ'

nll = []        # negative log likelihood

plots = []
for idx, mu in enumerate(mean_range):
    nll.append(-log_likelihood_gaussian(data, mu, fixed_sigma))
    print(mu, likelihood_gaussian(data, mu, fixed_sigma), nll[-1])

    @gif.frame
    def plot(idx):
        plt.hist(data, bins=60, density=True, alpha=0.6, color='g', label="Data")
        plt.plot(x, gaussian(x, mu, fixed_sigma), label=letters[idx])

        # Add title and labels
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.legend()


    plots.append(plot(idx))
    plots[-1].show()
    #plt.savefig(f'/home/moustafa/Desktop/{idx}.png')

# https://github.com/maxhumber/gif
#gif.save(plots, '/home/moustafa/Desktop/ex.gif', duration=1000)

plt.plot(mean_range, np.array(nll), marker='o')
plt.title("Mean vs Negative Log-likelihood")
plt.xlabel("mean")
plt.ylabel("nnl")
plt.show()

