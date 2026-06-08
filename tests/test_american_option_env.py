import numpy as np

from envs.american_option import AmericanOption


def test_american_option_reset():
    env = AmericanOption(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252)
    obs, info = env.reset(seed=0)
    assert obs.shape == (2,)
    assert obs[0] == 100
    assert obs[1] == 1.0
    assert env.path.shape == (1, 253)


def test_american_option_step_advances_time():
    env = AmericanOption(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252)
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0
    assert not terminated
    assert obs[1] < 1.0


def test_american_option_exercise_terminates():
    env = AmericanOption(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252)
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(1)
    assert terminated
    assert reward >= 0
