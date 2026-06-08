# Experiment 02: Ornstein-Uhlenbeck Path Simulation

Run: `python -m experiments.ou_paths.run`

## Setup

- X0=0, theta=2.0, mu=0.0, sigma=0.5, T=5.0
- 1,000 paths, 1,000 time steps
- 50 sample paths plotted

## Theory

The OU process replaces GBM's constant drift with a mean-reverting drift:

```
dX_t = theta * (mu - X_t) * dt + sigma * dW_t
```

Unlike GBM, OU has memory. The drift actively pulls the process back toward mu at rate theta. This prevents indefinite drift and produces a stationary distribution at long horizons with variance:

```
Var(X_inf) = sigma^2 / (2 * theta)
```

OU is used in finance to model interest rates, volatility, and mean-reverting spreads — any quantity anchored to a fundamental value.

## Results

OU sample paths


| Metric        | Simulated | Theoretical                        |
| ------------- | --------- | ---------------------------------- |
| Terminal mean | -0.009    | mu = 0.0                           |
| Terminal std  | 0.2503    | sqrt(sigma^2 / (2*theta)) = 0.2500 |
| Terminal var  | 0.0626    | sigma^2 / (2*theta) = 0.0625       |


## Analysis

The results confirm that the OU process has reached its stationary distribution by T=5.0. The terminal mean of -0.009 is within 1% of the theoretical long-run mean of 0.0 — indistinguishable from zero at this sample size. The terminal variance of 0.0626 matches the theoretical stationary variance of 0.0625 to three decimal places (0.16% error), confirming that theta=2.0 is strong enough to fully absorb the initial condition X0=0 well before T=5.0.

This is the defining contrast with GBM: where GBM variance grows without bound as sigma^2 * T, OU variance converges to a finite ceiling sigma^2 / (2*theta). At T=5.0, GBM with the same sigma would have variance sigma^2 * T = 1.25 — twenty times larger than the OU stationary variance of 0.0625. The mean-reverting drift acts as a restoring force that caps uncertainty regardless of the time horizon.

The speed of convergence to stationarity is governed by theta. At theta=2.0 the process has a half-life of ln(2)/theta ≈ 0.35 time units — meaning deviations from mu decay by half in roughly 0.35 units. By T=5.0, the process has completed over 14 half-lives, making the influence of X0 negligible.

## Open questions

- Estimate theta from simulated paths by regressing dX_t on (mu - X_t) and compare the recovered theta to the input value of 2.0
- At what T does the process effectively reach stationarity? Plot terminal variance vs T and mark the theoretical ceiling sigma^2/(2*theta) = 0.0625
- Contrast with GBM experiment 01: plot terminal variance vs T for both processes on the same axes to visualize bounded vs unbounded uncertainty
- How does the half-life ln(2)/theta change the path behavior? Re-run with theta=0.5 (slow reversion) and theta=10.0 (fast reversion)