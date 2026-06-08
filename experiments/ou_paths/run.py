import numpy as np
import matplotlib.pyplot as plt
from processes.ou import simulate_ou

X0, theta, mu, sigma, T, n_steps, n_paths = 0, 2.0, 0.0, 0.5, 5.0, 1000, 1000
X = simulate_ou(X0=X0, theta=theta, mu=mu, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths)
t = np.linspace(0, T, n_steps + 1)

# summary statistics
terminal = X[:, -1]
stationary_var = sigma**2 / (2 * theta)

print(f"Terminal mean: {np.mean(terminal):.4f}")
print(f"Terminal std: {np.std(terminal):.4f}")
print(f"Theoretical stationary std: {np.sqrt(stationary_var):.4f}")
print(f"Theoretical stationary var: {stationary_var:.4f}")
print(f"Simulated terminal var: {np.var(terminal):.4f}")

# plot
plt.figure(figsize=(12, 6))
plt.plot(t, X[:50].T, alpha=0.3, linewidth=0.8)
plt.axhline(y=mu, color='red', linewidth=1.5, linestyle='--', label='long-run mean')
plt.xlabel("Time")
plt.ylabel("X")
plt.title("Ornstein-Uhlenbeck — 50 paths")
plt.legend()
plt.tight_layout()
plt.savefig("experiments/ou_paths/ou_paths.png", dpi=150)
plt.show()