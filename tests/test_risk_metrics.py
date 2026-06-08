import numpy as np

from risk.metrics import cvar, var


def test_var_is_positive_loss():
    pnls = np.array([-2.0, -1.5, -0.5, 0.1, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0])
    assert var(pnls, alpha=0.95) > 0


def test_cvar_exceeds_var_for_fat_tail():
    pnls = np.array([-10.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert cvar(pnls, alpha=0.95) >= var(pnls, alpha=0.95)
