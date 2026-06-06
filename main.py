import matplotlib.pyplot as plt
import numpy as np
from processes.gbm import simulate_gbm
from processes.oh import simulate_ou
from envs.american_option import AmericanOption
from agents.dqn import train_dqn

"""
Simulate a Geometric Brownian Motion (GBM) and plot the paths. 

Graph displays possible stock prices under Black-Scholes assumptions:

About the stock price:
1. Price follows GBM
2. Drift and volatilty are constant over time
3. Volatility is the same regardless of strike or expiry

About the market:
1. No arbitrage
2. No dividends
3. Continous trading is possible at all times
4. You can buy/sell any fractional amount of shares
4. Risk-free rate is constant and known

About costs:
1. No transaction costs
2. No restricitons on short selling 
"""
"""
S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=1000, n_paths=1000)
t = np.linspace(0, 1.0, 1001)

plt.figure(figsize=(12, 6))
plt.plot(t, S[:50, :].T, alpha=0.3, linewidth=0.8)
plt.xlabel('Time (t)')
plt.ylabel('Stock Price (S_t)')
plt.title('GBM Simulation - 50 paths')
plt.show()
"""

"""
Simulate a Ornstein-Uhlenbeck process and plot the paths. 

Similar assumption as GBM, but with mean reversion.

The drift term is now theta * (mu - X) instead of mu, 
where theta is the mean reversion speed. 

OH introduces memory to the process, as the current value is influenced by the past values
and not just the current time step. This helps model more realistic processes with 
a fundamental value, preventing drifts to infinity.
"""

"""
X = simulate_ou(X0=0, theta=2.0, mu=0.0, sigma=0.5, T=5.0, n_steps=1000, n_paths=50)
t = np.linspace(0, 5.0, 1001)

plt.figure(figsize=(12, 6))
plt.plot(t, X[:50, :].T, alpha=0.3, linewidth=0.8)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='long-run mean')
plt.xlabel('Time (t)')
plt.ylabel('Process Value (X_t)')
plt.title('OH Simulation - 50 paths')
plt.legend()
plt.show()
"""
"""
Verify American Option environment works as expected.
"""

env = AmericanOption(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252)

"""
obs, info = env.reset()
print("Initial observation:", obs)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"observation: {obs}, reward: {reward:.2f}, terminated: {terminated}")
    if terminated:
        break
"""

model = train_dqn(env) # train DQN agent on the environment