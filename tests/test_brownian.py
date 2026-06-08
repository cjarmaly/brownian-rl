import numpy as np

from processes.brownian import simulate_brownian


def test_brownian_starts_at_zero():
    W = simulate_brownian(T=1, n_steps=100, n_paths=1000)
    assert W.shape == (1000, 101)
    assert np.all(W[:, 0] == 0)


def test_brownian_variance_scales_with_time():
    np.random.seed(42)
    W = simulate_brownian(T=1, n_steps=1000, n_paths=50_000)
    assert np.isclose(np.var(W[:, 500]), 0.5, atol=0.05)
    assert np.isclose(np.var(W[:, 1000]), 1.0, atol=0.05)
