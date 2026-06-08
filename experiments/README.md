# Experiments

Research scripts and written-up results. Each experiment has a `run.py` (code) and
`RESULTS.md` (findings and analysis).

## Tests

Fast correctness checks (no training):

```bash
pytest tests/ -m "not slow"
```

Slow tests (includes DQN/LSM if marked):

```bash
pytest tests/
```
