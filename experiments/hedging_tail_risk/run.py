"""Compute VaR/CVaR for RL vs BS hedging P&L distributions."""

import os

import matplotlib.pyplot as plt
import numpy as np

from agents.ppo import evaluate_hedging, train_ppo
from envs.hedging import HedgingEnv
from experiments.helpers import DEFAULT_HEDGE_PARAMS, run_bs_delta_hedge
from risk.metrics import cvar, var


def load_or_run_pnls(retrain=False):
    results_path = "experiments/ppo_hedging_vs_bs/results.npz"
    p = DEFAULT_HEDGE_PARAMS
    env = HedgingEnv(
        S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], n_steps=p["n_steps"]
    )

    if not retrain and os.path.exists(results_path):
        data = np.load(results_path)
        return data["rl_pnls"], data["bs_pnls"]

    model = train_ppo(env, total_timesteps=500_000)
    rl_pnls = evaluate_hedging(model, env, n_episodes=1000)
    bs_pnls = run_bs_delta_hedge(env, n_episodes=1000)
    np.savez(results_path, rl_pnls=rl_pnls, bs_pnls=bs_pnls)
    return rl_pnls, bs_pnls


def main():
    rl_pnls, bs_pnls = load_or_run_pnls(retrain=False)

    print("=== Hedging: RL Agent ===")
    print(f"VaR  95%: {var(rl_pnls):.4f}")
    print(f"CVaR 95%: {cvar(rl_pnls):.4f}")
    print(f"CVaR/VaR: {cvar(rl_pnls) / var(rl_pnls):.2f}")

    print("\n=== Hedging: Black-Scholes ===")
    print(f"VaR  95%: {var(bs_pnls):.4f}")
    print(f"CVaR 95%: {cvar(bs_pnls):.4f}")
    print(f"CVaR/VaR: {cvar(bs_pnls) / var(bs_pnls):.2f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(rl_pnls, bins=50, alpha=0.6, label="RL")
    plt.hist(bs_pnls, bins=50, alpha=0.6, label="BS")
    plt.axvline(-var(rl_pnls), color="blue", linestyle="--", label="RL VaR")
    plt.axvline(-var(bs_pnls), color="orange", linestyle="--", label="BS VaR")
    plt.title("Hedging P&L Distribution")
    plt.xlabel("P&L")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(rl_pnls, bins=50, alpha=0.6, label="RL")
    plt.hist(bs_pnls, bins=50, alpha=0.6, label="BS")
    plt.xlim(left=np.percentile(rl_pnls, 1))
    plt.title("Tail zoom")
    plt.xlabel("P&L")
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiments/hedging_tail_risk/tail_risk.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
