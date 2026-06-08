# Experiment 03: DQN on American Put
Run: `python -m experiments.dqn_american_put.run`

## Setup

- S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252
- DQN trained for 100k timesteps
- Evaluated over 1,000 episodes with deterministic policy
- Two baselines: always exercise immediately, always hold until expiry

## Results

|       Strategy       | Mean Payoff | Std  |
|        ---           |     ---     | ---  |
| Always exercise      | 0.0000      | 0.0000 |
| Always hold (expiry) | 6.4637      | 9.5668 |
| DQN agent            | 6.3077      | 9.3731 |

## Analysis

The always-exercise baseline returns exactly zero in every episode — at initiation the put is at-the-money (S0=K=100), so max(K-S0, 0) = 0 with certainty. This is not a failure of the baseline; it correctly reflects that immediate exercise of an at-the-money put has no intrinsic value. The time value of the option is strictly positive, so early exercise at t=0 is never rational.

The always-hold baseline (6.46) is a natural upper reference for a naive investor who never considers early exercise. Interestingly, the DQN agent (6.31) scores slightly below always-hold — a gap of 0.15, roughly 2.3%. This suggests the agent is exercising early on some paths where waiting would have been better, consistent with a policy that hasn't fully converged at 100k timesteps (ep_rew_mean=2.36 at end of training, well below the evaluation mean of 6.31).

The high std (9.37) across both DQN and always-hold reflects the natural skew of put payoffs: many episodes expire worthless (stock above strike), a few produce large payoffs (stock well below strike). This is not noise; instead, it is the correct shape of the payoff distribution.

The key question deferred to Experiment 04: does the DQN's learned early exercise policy recover value relative to the risk-neutral benchmark (LSM), or does suboptimal stopping destroy value?

## Open questions

- [ ] DQN scores 2.3% below always-hold — does this gap close with
      500k timesteps of training?