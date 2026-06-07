import matplotlib.pyplot as plt
import numpy as np
from processes.gbm import simulate_gbm
from processes.oh import simulate_ou
from envs.american_option import AmericanOption
from agents.dqn import train_dqn
from agents.ls import longstaff_schwartz
from envs.hedging import HedgingEnv
from agents.ppo import train_ppo
from agents.delta_hedge import bs_put_delta
from agents.delta_hedge import bs_put_price
from agents.ppo import evaluate_hedging
from agents.girsanov import radon_nikodym
from processes.brownian import simulate_brownian
from processes.gbm import simulate_gbm_from_brownian
from risk.metrics import var, cvar


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

# S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=1000, n_paths=1000)
# t = np.linspace(0, 1.0, 1001)

# plt.figure(figsize=(12, 6))
# plt.plot(t, S[:50, :].T, alpha=0.3, linewidth=0.8)
# plt.xlabel('Time (t)')
# plt.ylabel('Stock Price (S_t)')
# plt.title('GBM Simulation - 50 paths')
# plt.show()


"""
Simulate a Ornstein-Uhlenbeck process and plot the paths. 

Similar assumption as GBM, but with mean reversion.

The drift term is now theta * (mu - X) instead of mu, 
where theta is the mean reversion speed. 

OH introduces memory to the process, as the current value is influenced by the past values
and not just the current time step. This helps model more realistic processes with 
a fundamental value, preventing drifts to infinity.
"""


# X = simulate_ou(X0=0, theta=2.0, mu=0.0, sigma=0.5, T=5.0, n_steps=1000, n_paths=50)
# t = np.linspace(0, 5.0, 1001)

# plt.figure(figsize=(12, 6))
# plt.plot(t, X[:50, :].T, alpha=0.3, linewidth=0.8)
# plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='long-run mean')
# plt.xlabel('Time (t)')
# plt.ylabel('Process Value (X_t)')
# plt.title('OH Simulation - 50 paths')
# plt.legend()
# plt.show()

"""
Verify American Option environment works as expected.
"""

# shared parameters
# S0, K, T, r, sigma, n_steps, n_paths = 100, 100, 1.0, 0.05, 0.2, 252, 10000

# env = AmericanOption(S0=S0, K=K, T=T, r=r, sigma=sigma, n_steps=n_steps)


# obs, info = env.reset()
# print("Initial observation:", obs)

# for _ in range(10):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"observation: {obs}, reward: {reward:.2f}, terminated: {terminated}")
#     if terminated:
#         break


# model = train_dqn(env) # train DQN agent on the environment

"""
Evaluate the learned policy from the DQN agent.
"""
# n_eval = 1000
# rewards = []

# for _ in range(n_eval):
#     obs, _ = env.reset()
#     terminated = False
#     while not terminated:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#     rewards.append(reward)

# print(f"DQN mean payoff: {np.mean(rewards):.2f}")
# print(f"DQN std payoff: {np.std(rewards):.2f}")

"""
Compare DQN and LSM prices.
"""

# # simulate paths for LSM
# paths = simulate_gbm(S0=S0, mu=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths)

# # LSM price
# lsm_price = longstaff_schwartz(paths, K=K, r=r, dt=T/n_steps)
# print(f"LSM price:  {lsm_price:.4f}")


"""
Train a PPO agent to hedge a put option, and compare its performance to the Black-Scholes delta hedge.
"""

S0, K, T, r, sigma, n_steps = 100, 100, 1.0, 0.05, 0.2, 252

hedge_env = HedgingEnv(S0=S0, K=K, T=T, r=r, sigma=sigma, n_steps=n_steps)
model = train_ppo(hedge_env)
rl_pnls = evaluate_hedging(model, hedge_env, n_episodes=1000)

# BS delta hedge P&L — always pass bs_put_delta as action
bs_pnls = []
for _ in range(1000):
    obs, _ = hedge_env.reset()
    episode_pnl = 0
    terminated = False
    while not terminated:
        S, T_remaining, _ = obs
        action = np.array([bs_put_delta(S, K, T_remaining, r, sigma)])
        obs, reward, terminated, truncated, info = hedge_env.step(action)
        episode_pnl += reward
    bs_pnls.append(episode_pnl)
bs_pnls = np.array(bs_pnls)

print(f"RL  — mean P&L: {np.mean(rl_pnls):.4f}, std: {np.std(rl_pnls):.4f}")
print(f"BS  — mean P&L: {np.mean(bs_pnls):.4f}, std: {np.std(bs_pnls):.4f}")

# # Note that the BS delta hedge has much lower variance than the RL agent, indicating that the RL agent has not yet
# # converged on a strategy. This is expected, as the RL agent is learning a policy directly from the environment,
# # while the BS delta hedge is a deterministic strategy. However, the RL agent is able to achieve a higher mean P&L,
# # which (as indicated by the high std), indicates that it is taking incosistent positions which happen to average out better.
# # Also note that BS P&L is slightly negative— this is our gamma P&L, a result of rebalancing every day when the BS assumption assumes
# # continuous rebalancing. 

# # To improve our RL model, we can (1) train for longer or (2) penalize variance explicitly in the reward function

"""
Verify Girsanov's theorem works as expected.
"""

# mu, r, sigma, T = 0.1, 0.05, 0.2, 1.0
# n_paths = 100000

# W = simulate_brownian(T, n_steps=252, n_paths=n_paths)
# W_T = W[:, -1]  # terminal values

# # compute radon-nikodym weights
# L_T = radon_nikodym(mu, r, sigma, W_T, T)

# # property 1: E[L_T] should be 1.0
# print(f"E[L_T]: {np.mean(L_T):.4f}")

# # property 2: reweighted mean of W_T under Q
# # under P: E[W_T] = 0
# # under Q: E[W_T] = -lambda * T
# lambda_ = (mu - r) / sigma
# print(f"E[W_T] under P:          {np.mean(W_T):.4f}")
# print(f"E[W_T] under Q (approx): {np.mean(L_T * W_T):.4f}")
# print(f"E[W_T] under Q (exact):  {-lambda_ * T:.4f}")


"""
Compare pricing of an American option under the risk-neutral (Black Scholes) measure and the real-world (Girsanov) measure.
"""
# S0, K, T, mu, r, sigma, n_steps = 100, 100, 1.0, 0.1, 0.05, 0.2, 252
# n_paths = 100000

# # simulate W first, construct S from same W
# W = simulate_brownian(T, n_steps=252, n_paths=n_paths)
# S = simulate_gbm_from_brownian(W, S0=100, mu=mu, sigma=sigma, T=T)
# W_T = W[:, -1]

# payoffs = np.maximum(K - S[:, -1], 0)
# L_T = radon_nikodym(mu, r, sigma, W_T, T)

# price_P  = np.exp(-r * T) * np.mean(payoffs)
# price_Q  = np.exp(-r * T) * np.mean(L_T * payoffs)
# price_BS = bs_put_price(100, K, T, r, sigma)

# print(f"Price under P:        {price_P:.4f}")
# print(f"Price under Q:        {price_Q:.4f}")
# print(f"Black-Scholes price:  {price_BS:.4f}")

# # Under the real-world measure P, the stock drifts at mu=0.10.
# # Under the risk-neutral measure Q, it drifts at r=0.05.
# # The Radon-Nikodym derivative L_T bridges the two measures exactly.

# # Pricing under P (3.92) understates the true option price because
# # the high real-world drift makes put payoffs less likely on average.
# # Reweighting by L_T corrects for this, recovering the Q price (5.53)
# # which matches Black-Scholes (5.57) up to Monte Carlo error.


# # The drift mu never appears in Black-Scholes because Girsanov removes it.
# # The market price of risk lambda = (mu - r) / sigma quantifies what gets removed.

# # Our DQN and PPO agents approximate this reweighting implicitly
# # rather than computing it explicitly. DQN ignores the distribution
# # mismatch between old and new policies entirely, relying on the
# # replay buffer as a crude correction. PPO clips the policy update
# # to limit distribution shift, but never computes the exact weight.

"""
Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) for a given portfolio of P&L values.
"""

# pnls = evaluate_hedging(model, hedge_env)

# # American option (DQN)
# print("=== American Option (DQN) ===")
# print(f"VaR  95%: {var(pnls):.4f}")
# print(f"CVaR 95%: {cvar(pnls):.4f}")

# # Hedging (RL vs BS)
# print("\n=== Hedging: RL Agent ===")
# print(f"VaR  95%: {var(rl_pnls):.4f}")
# print(f"CVaR 95%: {cvar(rl_pnls):.4f}")

# print("\n=== Hedging: Black-Scholes ===")
# print(f"VaR  95%: {var(bs_pnls):.4f}")
# print(f"CVaR 95%: {cvar(bs_pnls):.4f}")

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.hist(rl_pnls, bins=50, alpha=0.6, label='RL')
# plt.hist(bs_pnls, bins=50, alpha=0.6, label='BS')
# plt.axvline(-var(rl_pnls), color='blue', linestyle='--', label='RL VaR')
# plt.axvline(-var(bs_pnls), color='orange', linestyle='--', label='BS VaR')
# plt.title("Hedging P&L Distribution")
# plt.xlabel("P&L")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.hist(rl_pnls, bins=50, alpha=0.6, label='RL')
# plt.hist(bs_pnls, bins=50, alpha=0.6, label='BS')
# plt.xlim(left=np.percentile(rl_pnls, 1))
# plt.title("Tail zoom")
# plt.xlabel("P&L")
# plt.legend()

# plt.tight_layout()
# plt.show()


# Black-Scholes keeps losses tightly bounded. CVaR barely exceeds
# VaR, meaning there is almost no tail beyond the 95th percentile.
# The RL agents have VaR five times larger and fat tails beyond it—
# bad episodes get significantly worse than the VaR threshold suggests.

# The CVaR/VaR ratio is the key signal. A ratio near 1.0 means the
# tail is thin. A ratio above 1.2 means
# the tail is fat (your worst case is worse than VaR implies).


# These results quantify exactly what it costs to use an RL agent
# that hasn't fully converged: five times the tail risk of the
# analytical benchmark. With more training, a variance-penalized
# reward, or explicit Girsanov reweighting, that gap would narrow.