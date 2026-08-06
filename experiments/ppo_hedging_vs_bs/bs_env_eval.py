import numpy as np
from envs.hedging import HedgingEnv
from experiments.helpers import DEFAULT_HEDGE_PARAMS

P = DEFAULT_HEDGE_PARAMS
S0, K, T, R, SIGMA, N_STEPS = (
    P["S0"], P["K"], P["T"], P["r"], P["sigma"], P["n_steps"],
)
DT = T / N_STEPS
N_EPISODES = 2000


def main():
    totals = []
    env = HedgingEnv(S0, K, T, R, SIGMA, N_STEPS)
    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        total = 0
        terminated = False
        while not terminated:
            action = [obs[2]]
            obs, reward, terminated, _, _ = env.step(action)
            total += reward
        totals += [total]

    print(f"Mean: {np.mean(totals)}; STD: {np.std(totals)}")
    
if __name__ == "__main__":
    main()