"""P&L attribution for Experiment 05: where does the RL agent's mean 'edge' come from?

Run: `python -m experiments.ppo_hedging_vs_bs.attribution`

No agent checkpoint is saved by run.py, so the mechanism is tested with
alpha-scaled BS hedges (h = alpha * bs_put_delta, alpha from 1 down to 0),
under the exact accounting of envs/hedging.py. alpha=1.0 reproduces the saved
bs_pnls in results.npz, validating the harness; alpha=0.0 (no hedging at all)
turns out to reproduce the trained agent's profile.

Findings (Aug 5, 2026):
1. The trained PPO agent (-0.37, 9.08) is statistically indistinguishable from
   not hedging (-0.25, 8.94). The agent learned ~"don't hedge."
2. Cause: the env reward pays no interest on short-stock proceeds, so hedging
   bleeds the risk-neutral drift uncompensated (~ h*S*r ~ -2/yr — the entire
   BS hedge "cost"). Add the cash-account term and the full hedge goes to
   (-0.29, 0.50): free, and ~18x less P&L variance than the agent.
3. Residual exposure is directional, not gamma: episode P&L regresses on
   terminal return with R^2 ~ 0.6 and on realized variance with R^2 ~ 0.00.
   (mu = r in the env, so there is no mu - r premium to capture by design.)
"""

import math

import numpy as np
from scipy.special import ndtr

from experiments.helpers import DEFAULT_HEDGE_PARAMS

P = DEFAULT_HEDGE_PARAMS
S0, K, T, R, SIGMA, N_STEPS = (
    P["S0"], P["K"], P["T"], P["r"], P["sigma"], P["n_steps"],
)
DT = T / N_STEPS
N_EPISODES = 2000


# Vectorized mirrors of agents.delta_hedge (those are scalar; these take arrays).
def put_delta(S, tau):
    d1 = (np.log(S / K) + (R + 0.5 * SIGMA**2) * tau) / (SIGMA * math.sqrt(tau))
    return ndtr(d1) - 1.0


def put_price(S, tau):
    if tau <= 0:
        return np.maximum(K - S, 0.0)
    sq = SIGMA * math.sqrt(tau)
    d1 = (np.log(S / K) + (R + 0.5 * SIGMA**2) * tau) / sq
    d2 = d1 - sq
    return K * math.exp(-R * tau) * ndtr(-d2) - S * ndtr(-d1)


def simulate_paths(n, rng):
    """GBM with mu = r, matching envs/hedging.py."""
    z = rng.standard_normal((n, N_STEPS))
    increments = (R - 0.5 * SIGMA**2) * DT + SIGMA * math.sqrt(DT) * z
    log_paths = np.cumsum(increments, axis=1)
    return S0 * np.exp(np.hstack([np.zeros((n, 1)), log_paths]))


def run_hedge(paths, alpha, financing=False):
    """Episode P&L for h_t = alpha * put_delta with env-identical accounting.

    financing=True adds the cash-account term the env omits: interest on the
    short-stock proceeds (-h * S earns r each step).
    """
    pnls = np.zeros(paths.shape[0])
    for t in range(N_STEPS):
        tau = T - t * DT
        s_prev, s_curr = paths[:, t], paths[:, t + 1]
        h = alpha * put_delta(s_prev, tau)
        pnls += h * (s_curr - s_prev)
        pnls -= put_price(s_curr, tau - DT) - put_price(s_prev, tau)
        if financing:
            pnls += (-h * s_prev) * (math.exp(R * DT) - 1.0)
    return pnls


def ols(y, x):
    x1 = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(x1, y, rcond=None)
    r2 = 1 - (y - x1 @ beta).var() / y.var()
    return beta[1], r2


def main():
    rng = np.random.default_rng(42)
    paths = simulate_paths(N_EPISODES, rng)
    term_ret = paths[:, -1] / S0 - 1.0
    realized_var = (np.diff(np.log(paths), axis=1) ** 2).sum(axis=1)

    print(f"{'alpha':>6} {'financing':>9} {'mean':>9} {'std':>8}")
    books = {}
    for alpha in (1.0, 0.8, 0.5, 0.2, 0.0):
        books[alpha] = run_hedge(paths, alpha)
        print(
            f"{alpha:>6} {'no':>9} "
            f"{books[alpha].mean():>9.4f} {books[alpha].std():>8.4f}"
        )
    fin = run_hedge(paths, 1.0, financing=True)
    print(f"{1.0:>6} {'YES':>9} {fin.mean():>9.4f} {fin.std():>8.4f}")

    d = np.load("experiments/ppo_hedging_vs_bs/results.npz")
    print("\nSaved results.npz:")
    for k in d.files:
        print(f"  {k}: mean {d[k].mean():.4f}, std {d[k].std():.4f}")

    for alpha in (0.5, 1.0):
        print(f"\nP&L attribution regressions (alpha = {alpha} book):")
        for name, x in (("terminal return", term_ret), ("realized variance", realized_var)):
            slope, r2 = ols(books[alpha], x)
            print(f"  P&L ~ {name:<18} slope {slope:>10.3f}   R^2 {r2:.3f}")


if __name__ == "__main__":
    main()
