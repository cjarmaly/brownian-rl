# Experiment 04: DQN vs Longstaff-Schwartz
Run: `python -m experiments.dqn_vs_lsm.run`

## Setup

- S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252
- LSM uses 10,000 simulated GBM paths under risk-neutral drift r=0.05
- DQN trained for 100k timesteps, evaluated over 1,000 episodes

## Results

|    Method    | Price  |
|     ---      |  ---   |
| DQN agent    | 5.6173 |
| LSM (benchmark) | 6.0224 |
| Gap          | 0.4051 (6.7%) |

## Analysis

The DQN agent recovers 93.3% of the LSM benchmark price, with a gap of $0.41. LSM is the industry-standard Monte Carlo method for American option pricing — it uses regression over 10,000 risk-neutral paths to estimate the optimal continuation value at each step. The DQN, by contrast, learns purely from experience with no access to the analytical structure of the problem.

The 6.7% gap is attributable to two factors. First, 100k training timesteps is insufficient for full convergence — ep_rew_mean of 0.527 at end of training is well below the evaluation mean of 5.62, indicating the agent was still learning at termination. Second, the DQN's early exercise policy is suboptimal on some paths: ep_len_mean of 8.91 steps (out of 252) suggests the agent exercises very early on average, foregoing time value on paths where waiting would have been more profitable.

Notably, the gap varies across runs due to the stochastic nature of both methods — LSM carries its own Monte Carlo error (roughly ±0.1 at 10,000 paths) and DQN payoffs have high variance (std ~9.3 from Experiment 03). The 6.7% gap should therefore be interpreted as an upper bound on true suboptimality — with more training timesteps and more evaluation episodes, it narrows.

## Correction (2026-08-02, code review)

The attribution above is partly wrong. Re-deriving the discount structure exposed two
specification bugs that confound the comparison:

1. **The DQN's discount factor is mis-set.** Training used `gamma=0.99` *per step* at 252
   steps/year, an implied continuously-compounded rate of −252·ln(0.99) ≈ **2.53/yr — roughly
   50× the model's r = 0.05**. The correct per-step factor is e^(−r·dt) ≈ 0.99980. The
   "very early exercise" flagged above (ep_len_mean = 8.91) is therefore not evidence of
   undertraining: the agent was *optimally* solving a problem in which waiting costs ~1% per
   day. The policy learned the objective it was given; the objective was wrong.

2. **The evaluation metric and the benchmark are different objects.** The eval loop appends
   raw undiscounted payoffs, E[payoff(τ)], while LSM reports the discounted price
   E[e^(−rτ)·payoff(τ)]. With r = 0.05 and T = 1, that wedge can be up to ~5% by itself
   (smaller here because exercise happens early).

Until both are fixed, the headline "DQN recovers 93.3% of LSM" is not interpretable as a
measure of policy suboptimality.

## Open questions

- [ ] Fix `gamma = exp(-r*dt)` in `agents/dqn.py` and retrain — does the early-exercise
      pathology (ep_len_mean ≈ 9) disappear?
- [ ] Discount eval rewards by e^(−r·τ·dt) in `experiments/dqn_vs_lsm/run.py` so both
      columns of the table are prices under the same measure and numeraire.
- [ ] Then rerun the original question: train for 500k timesteps and measure gap reduction —
      does it converge toward LSM as predicted?
