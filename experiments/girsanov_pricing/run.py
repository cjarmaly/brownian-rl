"""Verify Girsanov measure change and compare P vs Q put pricing."""

import numpy as np

from agents.delta_hedge import bs_put_price
from agents.girsanov import radon_nikodym
from processes.brownian import simulate_brownian
from processes.gbm import simulate_gbm_from_brownian


def verify_girsanov(mu=0.1, r=0.05, sigma=0.2, T=1.0, n_paths=100_000):
    W = simulate_brownian(T, n_steps=252, n_paths=n_paths)
    W_T = W[:, -1]
    L_T = radon_nikodym(mu, r, sigma, W_T, T)
    lambda_ = (mu - r) / sigma

    print("=== Girsanov verification ===")
    print(f"E[L_T]:                  {np.mean(L_T):.4f}  (expect ~1.0)")
    print(f"E[W_T] under P:          {np.mean(W_T):.4f}  (expect ~0.0)")
    print(f"E[W_T] under Q (approx): {np.mean(L_T * W_T):.4f}")
    print(f"E[W_T] under Q (exact):  {-lambda_ * T:.4f}")


def compare_pricing(S0=100, K=100, T=1.0, mu=0.1, r=0.05, sigma=0.2, n_paths=100_000):
    W = simulate_brownian(T, n_steps=252, n_paths=n_paths)
    S = simulate_gbm_from_brownian(W, S0=S0, mu=mu, sigma=sigma, T=T)
    W_T = W[:, -1]

    payoffs = np.maximum(K - S[:, -1], 0)
    L_T = radon_nikodym(mu, r, sigma, W_T, T)

    price_P = np.exp(-r * T) * np.mean(payoffs)
    price_Q = np.exp(-r * T) * np.mean(L_T * payoffs)
    price_BS = bs_put_price(S0, K, T, r, sigma)

    print("\n=== P vs Q pricing ===")
    print(f"Price under P:        {price_P:.4f}")
    print(f"Price under Q:        {price_Q:.4f}")
    print(f"Black-Scholes price:  {price_BS:.4f}")


def main():
    verify_girsanov()
    compare_pricing()


if __name__ == "__main__":
    main()
