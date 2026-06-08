"""Plot sample GBM paths under Black-Scholes assumptions."""

import numpy as np
import matplotlib.pyplot as plt
from processes.gbm import simulate_gbm

S0, mu, sigma, T, n_steps, n_paths = 100, 0.05, 0.2, 1.0, 1000, 1000
S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths)
t = np.linspace(0, T, n_steps + 1)

# summary statistics
terminal = S[:, -1]
log_returns = np.log(terminal / S0)

print(f"Terminal price mean:      {np.mean(terminal):.4f}")
print(f"Terminal price std:       {np.std(terminal):.4f}")
print(f"Theoretical mean:         {S0 * np.exp(mu * T):.4f}")
print(f"Log-return mean:          {np.mean(log_returns):.4f}")
print(f"Log-return std:           {np.std(log_returns):.4f}")
print(f"Theoretical log-ret mean: {(mu - 0.5 * sigma**2) * T:.4f}")
print(f"Theoretical log-ret std:  {sigma * np.sqrt(T):.4f}")

# plot
plt.figure(figsize=(12, 6))
plt.plot(t, S[:50].T, alpha=0.3, linewidth=0.8)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("GBM — 50 sample paths")
plt.tight_layout()
plt.savefig("experiments/gbm_paths/gbm_paths.png", dpi=150)
plt.show()