import matplotlib.pyplot as plt
import numpy as np
from processes.gbm import simulate_gbm

S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=1000, n_paths=1000)
t = np.linspace(0, 1.0, 1001)

plt.figure(figsize=(12, 6))
plt.plot(t, S[:50, :].T, alpha=0.3, linewidth=0.8)
plt.xlabel('Time (t)')
plt.ylabel('Stock Price (S_t)')
plt.title('GBM Simulation - 50 paths')
plt.show()