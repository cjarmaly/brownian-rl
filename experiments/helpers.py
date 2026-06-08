"""Shared helpers for experiment scripts."""

import numpy as np

from agents.delta_hedge import bs_put_delta

DEFAULT_HEDGE_PARAMS = {
    "S0": 100,
    "K": 100,
    "T": 1.0,
    "r": 0.05,
    "sigma": 0.2,
    "n_steps": 252,
}

DEFAULT_OPTION_PARAMS = {
    **DEFAULT_HEDGE_PARAMS,
    "n_paths": 10_000,
}


def run_bs_delta_hedge(env, n_episodes=1000):
    """Roll out Black-Scholes delta hedging for n episodes."""
    K = env.K
    r = env.r
    sigma = env.sigma
    pnls = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_pnl = 0
        terminated = False
        while not terminated:
            S, T_remaining, _ = obs
            action = np.array([bs_put_delta(S, K, T_remaining, r, sigma)], dtype=np.float32)
            obs, reward, terminated, _, _ = env.step(action)
            episode_pnl += reward
        pnls.append(episode_pnl)
    return np.array(pnls)
