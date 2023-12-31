import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm

# Parameters for the log-normal distribution
mu = 2  # Mean of the underlying normal distribution
sigma = 1  # Standard deviation of the underlying normal distribution

# Generate samples from the log-normal distribution
y_samples = np.random.lognormal(mean=mu, sigma=sigma, size=10000)

# Apply the transformation X = exp(-Y)
x_samples = np.exp(-y_samples)

# Plotting the PDF of X
plt.figure(figsize=(10, 6))
sns.histplot(x_samples, bins=50, kde=True, stat="density", log_scale=(True, False))
plt.xlabel('X (log scale)')
plt.xlim((1e-3, 1))
plt.ylabel('Probability Density')
plt.title('PDF of X = exp(-Y), Y ~ LogNormal')
plt.grid(True)
plt.show()
