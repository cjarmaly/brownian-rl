# Experiments

Research scripts and written-up results. Each experiment has a `run.py` (code) and
`RESULTS.md` (findings and analysis).

## Quick reference

| # | Experiment | Command | Status |
|---|---|---|---|
| 01 | GBM paths | `python -m experiments.gbm_paths.run` | Needs results |
| 02 | OU paths | `python -m experiments.ou_paths.run` | Needs results |
| 03 | DQN American put | `python -m experiments.dqn_american_put.run` | Needs results |
| 04 | DQN vs LSM | `python -m experiments.dqn_vs_lsm.run` | Needs results |
| 05 | PPO vs BS hedge | `python -m experiments.ppo_hedging_vs_bs.run` | Needs results |
| 06 | Girsanov pricing | `python -m experiments.girsanov_pricing.run` | Partial analysis |
| 07 | Hedging tail risk | `python -m experiments.hedging_tail_risk.run` | Partial analysis |

Run from the project root with the venv active.

## Where analysis is still needed

Sections marked `<!-- TODO -->` in each `RESULTS.md` need numbers filled in after
you run the experiment. Priority:

1. **05 + 07** — hedging comparison and tail risk (related; 07 loads 05's `results.npz`)
2. **06** — Girsanov pricing (verification + P vs Q table)
3. **04** — DQN vs LSM benchmark gap
4. **03** — DQN payoff statistics
5. **01 + 02** — visual/process validation plots

## Tests

Fast correctness checks (no training):

```bash
pytest tests/ -m "not slow"
```

Slow tests (includes DQN/LSM if marked):

```bash
pytest tests/
```
