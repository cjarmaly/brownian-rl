import numpy as np

from agents.girsanov import radon_nikodym
from processes.brownian import simulate_brownian


def test_radon_nikodym_expectation_is_one():
    np.random.seed(42)
    W = simulate_brownian(T=1.0, n_steps=252, n_paths=100_000)
    L_T = radon_nikodym(mu=0.1, r=0.05, sigma=0.2, W_T=W[:, -1], T=1.0)
    assert np.isclose(np.mean(L_T), 1.0, atol=0.01)


def test_radon_nikodym_changes_drift():
    np.random.seed(42)
    mu, r, sigma, T = 0.1, 0.05, 0.2, 1.0
    W = simulate_brownian(T=T, n_steps=252, n_paths=100_000)
    W_T = W[:, -1]
    L_T = radon_nikodym(mu, r, sigma, W_T, T)
    lambda_ = (mu - r) / sigma
    assert np.isclose(np.mean(L_T * W_T), -lambda_ * T, atol=0.05)
