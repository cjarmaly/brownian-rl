import numpy as np

from agents.delta_hedge import bs_put_price
from agents.ls import longstaff_schwartz
from processes.gbm import simulate_gbm


def test_lsm_price_is_reasonable():
    np.random.seed(0)
    paths = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=50_000)
    lsm_price = longstaff_schwartz(paths, K=100, r=0.05, dt=1.0 / 252)
    bs_price = bs_put_price(100, 100, 1.0, 0.05, 0.2)
    assert 0 < lsm_price < 100
    assert np.isclose(lsm_price, bs_price, atol=1.0)
