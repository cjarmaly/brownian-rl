"""Compare DQN policy payoffs to Longstaff-Schwartz (LSM) price."""

import numpy as np

from agents.dqn import train_dqn
from agents.ls import longstaff_schwartz
from envs.american_option import AmericanOption
from experiments.helpers import DEFAULT_OPTION_PARAMS
from processes.gbm import simulate_gbm


def main():
    p = DEFAULT_OPTION_PARAMS
    env = AmericanOption(
        S0=p["S0"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], n_steps=p["n_steps"]
    )

    model = train_dqn(env, total_timesteps=100_000)

    rewards = []
    for _ in range(1000):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)

    paths = simulate_gbm(
        S0=p["S0"], mu=p["r"], sigma=p["sigma"], T=p["T"],
        n_steps=p["n_steps"], n_paths=p["n_paths"],
    )
    lsm_price = longstaff_schwartz(paths, K=p["K"], r=p["r"], dt=p["T"] / p["n_steps"])

    print(f"DQN mean payoff: {np.mean(rewards):.4f}")
    print(f"LSM price:       {lsm_price:.4f}")


if __name__ == "__main__":
    main()
