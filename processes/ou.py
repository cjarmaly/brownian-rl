import numpy as np

"""
Simulate Ornstein-Uhlenbeck process.
"""

def simulate_ou(X0, theta, mu, sigma, T, n_steps, n_paths):
    """
    X0: initial value
    theta: mean reversion speed
    mu: long-run mean
    sigma: volatility
    returns: (n_paths, n_steps + 1) array of X_t values
    """

    dt = T / n_steps
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = X0
    for t in range(n_steps):
        X[:, t+1] = X[:, t] + theta * (mu - X[:, t]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, n_paths)
    return X
