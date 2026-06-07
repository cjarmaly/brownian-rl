import numpy as np

"""
Girsanov's theorem allows us to transform a probability measure to a new probability measure that is equivalent to the original measure.
This is useful for pricing options in a different probability measure, such as the risk-neutral measure.
"""

def radon_nikodym(mu, r, sigma, W_T, T):
    """
    Radon-Nikodym derivative of P with respect to Q.

    mu: drift
    r: risk-free rate
    sigma: volatility
    W_T: Brownian motion at time T
    T: time horizon
    returns: Scalar weight L_T
    """
    lambda_ = (mu - r) / sigma
    L_T = np.exp(-lambda_ * W_T - 0.5 * lambda_**2 * T)
    return L_T