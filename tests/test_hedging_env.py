import numpy as np

from agents.delta_hedge import bs_put_delta
from envs.hedging import HedgingEnv


def test_hedging_env_reset():
    env = HedgingEnv(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3,)
    assert obs[0] == 100
    assert obs[1] == 1.0
    assert -1 <= obs[2] <= 0


def test_hedging_env_runs_full_episode():
    env = HedgingEnv(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252)
    obs, _ = env.reset(seed=0)
    total_reward = 0
    terminated = False
    while not terminated:
        S, T_remaining, _ = obs
        action = np.array([bs_put_delta(S, 100, T_remaining, 0.05, 0.2)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    assert np.isfinite(total_reward)
