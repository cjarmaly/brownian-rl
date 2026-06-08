import pytest

S0 = 100
K = 100
T = 1.0
R = 0.05
SIGMA = 0.2
N_STEPS = 252
N_PATHS = 10_000


@pytest.fixture
def option_params():
    return {
        "S0": S0,
        "K": K,
        "T": T,
        "r": R,
        "sigma": SIGMA,
        "n_steps": N_STEPS,
        "n_paths": N_PATHS,
    }
