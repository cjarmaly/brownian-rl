# Experiment 06: Girsanov Measure Change and P vs Q Pricing
Run: `python -m experiments.girsanov_pricing.run`

## Setup

- S0=100, K=100, T=1.0, mu=0.10, r=0.05, sigma=0.2
- 100,000 Monte Carlo paths
- European put payoffs at expiry
- Same Brownian paths W used for both P and Q pricing via shared simulation

## Girsanov verification results

| Quantity | Simulated | Theoretical |
|---|---|---|
| E[L_T] | 1.0005 | 1.0000 |
| E[W_T] under P | -0.0016 | 0.0000 |
| E[W_T] under Q (approx) | -0.2523 | -0.2500 |
| E[W_T] under Q (exact) | -0.2500 | -0.2500 |

## Pricing results

| Measure | Price | Gap vs BS |
|---|---|---|
| P (real-world) | 3.9569 | -28.9% |
| Q (risk-neutral, reweighted) | 5.5897 | +0.3% |
| Black-Scholes | 5.5735 | — |

## Analysis

**Girsanov verification:**
All four quantities land within 1% of their theoretical values at 100,000 paths. E[L_T]=1.0005 confirms L_T is a valid probability measure — weights integrate to 1 up to Monte Carlo noise that shrinks as 1/sqrt(n_paths). The reweighted mean E[W_T] under Q (-0.2523) matches the exact value (-0.2500) to within 0.9%, verifying that multiplication by L_T correctly shifts the distribution from P to Q.

**Pricing:**
Pricing under the real-world measure P (3.96) understates the true option price by 28.9%. With mu=0.10 > r=0.05, the stock has higher real-world drift than the risk-neutral rate — paths trend upward more aggressively, put payoffs are less likely on average, and the naive P expectation underestimates fair value. This is precisely why P cannot be used for pricing: it conflates the drift of the underlying with the fair value of the derivative.

Reweighting by L_T recovers the Q price (5.59), matching Black-Scholes (5.57) to within 0.3% — well within Monte Carlo error at 100,000 paths. The residual gap shrinks as 1/sqrt(n_paths) and would reach ~0.1% at 1,000,000 paths.

The market price of risk lambda = (mu - r) / sigma = 0.25 quantifies the drift removed by Girsanov. A higher lambda means a larger wedge between P and Q prices — markets with high risk premia require larger measure corrections.

**Connection to our RL agents:**
The DQN and PPO agents approximate this reweighting implicitly. DQN uses a replay buffer to smooth over distribution shift between old and new policies — a crude analogue of importance weighting. PPO clips policy updates to bound the distribution shift, but never computes the exact Radon-Nikodym weight. A principled implementation would multiply each trajectory's return by L_T before updating — recovering the exact Girsanov correction in discrete time.

## Open questions

- [ ] At 100k paths, Monte Carlo error on Q price is ~0.3% — verify
      this shrinks proportionally to 1/sqrt(n_paths) at 10k, 100k, 1M
- [ ] Extend to American put: LSM under P vs LSM with Girsanov
      reweighting — does the same 28.9% gap appear?
