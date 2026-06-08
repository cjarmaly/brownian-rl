"""Train PPO hedging agent and compare to Black-Scholes delta hedge."""

import numpy as np

from agents.ppo import evaluate_hedging, train_ppo
from envs.hedging import HedgingEnv
from experiments.helpers import DEFAULT_HEDGE_PARAMS, run_bs_delta_hedge


def main():
    p = DEFAULT_HEDGE_PARAMS
    env = HedgingEnv(
        S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], n_steps=p["n_steps"]
    )

    model = train_ppo(env, total_timesteps=500_000)
    rl_pnls = evaluate_hedging(model, env, n_episodes=1000)
    bs_pnls = run_bs_delta_hedge(env, n_episodes=1000)

    print(f"RL  — mean P&L: {np.mean(rl_pnls):.4f}, std: {np.std(rl_pnls):.4f}")
    print(f"BS  — mean P&L: {np.mean(bs_pnls):.4f}, std: {np.std(bs_pnls):.4f}")

    np.savez(
        "experiments/ppo_hedging_vs_bs/results.npz",
        rl_pnls=rl_pnls,
        bs_pnls=bs_pnls,
    )
    print("Saved results to experiments/ppo_hedging_vs_bs/results.npz")


if __name__ == "__main__":
    main()
