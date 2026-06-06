import numpy as np
from .brownian import simulate_brownian

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths):
    """
    S0: initial stock price
    mu: drift
    sigma: volatility
    T: time horizon
    returns: (n_paths, n_steps + 1) array of S_t values
    """
    W = simulate_brownian(T, n_steps, n_paths)
    t = np.linspace(0, T, n_steps +1)
    return S0 * np.exp((mu - 0.5 * sigma**2) * t.reshape(1, -1) + sigma * W)

if __name__ == "__main__":
    S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1, n_steps=100, n_paths=1000)
    
    # Check all paths start at S0
    print(S[:, 0])

    # Check prices are always positive
    print (np.all(S > 0))

    # Check expected price at T should be ~S0 * exp(mu * T)
    expected_price = 100 * np.exp(0.05)
    print(np.mean(S[:, -1]), expected_price)