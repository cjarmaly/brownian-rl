import numpy as np

from processes.brownian import simulate_brownian
from processes.gbm import simulate_gbm, simulate_gbm_from_brownian


def test_gbm_shape_and_positive():
    S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=100, n_paths=100)
    assert S.shape == (100, 101)
    assert np.all(S > 0)


def test_gbm_starts_at_S0():
    S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=100, n_paths=10)
    assert np.allclose(S[:, 0], 100)


def test_gbm_from_brownian_shape_and_starting_price():
    np.random.seed(0)
    W = simulate_brownian(T=1.0, n_steps=100, n_paths=5)
    S = simulate_gbm_from_brownian(W, S0=100, mu=0.05, sigma=0.2, T=1.0)
    assert S.shape == W.shape
    assert np.allclose(S[:, 0], 100)
    assert np.all(S > 0)
