"""Train DQN on American put environment and evaluate policy payoffs."""

import numpy as np

from agents.dqn import train_dqn
from envs.american_option import AmericanOption
from experiments.helpers import DEFAULT_OPTION_PARAMS


def main():
    p = DEFAULT_OPTION_PARAMS
    # base case 1: always exercise immediately
    env = AmericanOption(
        S0=p["S0"], 
        K=p["K"], 
        T=p["T"], 
        r=p["r"], 
        sigma=p["sigma"], 
        n_steps=p["n_steps"],
        n_paths=p["n_paths"]
        )

    always_exercise = []
    for _ in range(1000):
        obs, _ = env.reset()
        _, reward, _, _, _ = env.step(1)
        always_exercise.append(reward)

    # base case 2: always hold until expiry
    always_hold = []
    for _ in range(1000):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            obs, reward, terminated, _, _ = env.step(0)
        always_hold.append(reward)

    model = train_dqn(env, total_timesteps=100_000)

    n_eval = 1000
    rewards = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)

    rewards = np.array(rewards)
    print(f"DQN mean payoff: {np.mean(rewards):.4f}")
    print(f"DQN std payoff: {np.std(rewards):.4f}")
    print(f"Always exercise mean: {np.mean(always_exercise):.4f}")
    print(f"Always hold mean: {np.mean(always_hold):.4f}")
    print(f"Always exercise std: {np.std(always_exercise):.4f}")
    print(f"Always hold std: {np.std(always_hold):.4f}")


if __name__ == "__main__":
    main()
