import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Values between 0 and 1 (inclusive)
x = np.linspace(0, 1, 1000)

# Different sets of alpha and beta parameters
params = [(50, 50), (1, 40), (51,89)]

# Create the plot
plt.figure(figsize=(12, 8))

for alpha, beta_ in params:
    y = beta.pdf(x, alpha, beta_)
    plt.plot(x, y, label=f'Beta({alpha},{beta_})')

plt.title('Beta Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
