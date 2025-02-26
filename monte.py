import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic data
def generate_data(n, beta_0, beta_1, noise_std):
    x = np.random.uniform(0, 10, n)
    error = np.random.normal(0, noise_std, n)
    y = beta_0 + beta_1 * x + error
    return x, y

# Function to perform simple linear regression
def simple_linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    beta_0 = y_mean - beta_1 * x_mean
    return beta_0, beta_1

# Monte Carlo simulation
def monte_carlo_simulation(num_simulations, n, beta_0, beta_1, noise_std):
    beta_0_estimates = []
    beta_1_estimates = []
    
    for _ in range(num_simulations):
        x, y = generate_data(n, beta_0, beta_1, noise_std)
        beta_0_hat, beta_1_hat = simple_linear_regression(x, y)
        beta_0_estimates.append(beta_0_hat)
        beta_1_estimates.append(beta_1_hat)
    
    return beta_0_estimates, beta_1_estimates

# Parameters
num_simulations = 1000
n = 100
true_beta_0 = 2.0
true_beta_1 = 3.0
noise_std = 1.0

# Run Monte Carlo simulation
beta_0_estimates, beta_1_estimates = monte_carlo_simulation(num_simulations, n, true_beta_0, true_beta_1, noise_std)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(beta_0_estimates, bins=30, edgecolor='k', alpha=0.7)
plt.axvline(true_beta_0, color='r', linestyle='dashed', linewidth=2)
plt.title('Distribution of Intercept Estimates')
plt.xlabel('Intercept')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(beta_1_estimates, bins=30, edgecolor='k', alpha=0.7)
plt.axvline(true_beta_1, color='r', linestyle='dashed', linewidth=2)
plt.title('Distribution of Slope Estimates')
plt.xlabel('Slope')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
