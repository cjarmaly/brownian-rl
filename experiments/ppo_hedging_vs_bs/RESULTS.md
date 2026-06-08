# Experiment 05: PPO Hedging vs Black-Scholes Delta
Run: `python -m experiments.ppo_hedging_vs_bs.run`

## Setup

- S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252
- PPO trained for 500k timesteps
- 1,000 evaluation episodes for both RL and BS delta hedge

## Results

|    Strategy    | Mean P&L | Std    |
|      ---       |   ---    |  ---   |
| PPO RL agent   | -0.3667  | 9.0849 |
| BS delta hedge | -2.1528  | 1.2381 |

Results saved to `experiments/ppo_hedging_vs_bs/results.npz` for use in Experiment 07.

## Analysis

The results reveal a fundamental tradeoff between mean P&L and variance. The BS delta hedge achieves a std of 1.24 — tight, consistent, and predictable. The RL agent's std of 9.08 is seven times larger, meaning its P&L swings wildly across episodes.

The BS mean P&L of -2.15 reflects gamma P&L — the cost of discrete daily rebalancing when Black-Scholes assumes continuous hedging. Each day the hedge is slightly misaligned, and these small errors accumulate over 252 steps. This is not a failure of the strategy; it is the irreducible cost of operating in discrete time under a continuous-time model.

The RL agent's mean P&L of -0.37 is closer to zero, but this is misleading. The agent is not hedging more precisely — it is taking inconsistent positions that happen to average out better across 1,000 episodes. The high std confirms this: some episodes produce large gains, others large losses, and the mean flatters the agent's true performance.

The training metrics support this interpretation. At 500k timesteps, explained_variance=0.107 — the value function explains only 10.7% of return variance, indicating the agent has not learned a reliable model of future rewards. The policy std of 0.256 (down from 0.424 at 200k) shows the agent is still narrowing its hedge ratio range but has not converged to a tight policy.

The correct metric for hedging quality is not mean P&L but P&L variance — a perfect hedge has zero variance regardless of mean. By this measure, BS outperforms the RL agent by a factor of seven. Tail risk implications are quantified in Experiment 07.

## Next steps

1. Train for 1M+ timesteps and track explained_variance as convergence signal
2. Modify reward to penalize variance explicitly: 
    reward = P&L - lambda * |P&L| to incentivize consistency over mean
3. Add transaction costs — BS loses its advantage under realistic costs, which is where RL becomes genuinely competitive
4. See Experiment 07 for VaR/CVaR comparison

## Open questions

- [ ] Does RL std converge toward BS std (1.24) with more training?
- [ ] At what transaction cost level does the RL agent's higher mean
      P&L offset its higher variance relative to BS?
- [ ] explained_variance=0.107 after 500k steps is low — would a larger
      network (e.g., 256x256 vs 64x64) improve convergence?