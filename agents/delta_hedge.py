import numpy as np
from scipy.stats import norm

def bs_put_delta(S, K, T, r, sigma):
    """
    Black-Scholes formula for Europena put option delta.

    Deep in the money, a put behaves like a short, so delta is -1.
    Deep out of the money, a put is worthless, so delta is 0.
    At the money, delta is -0.5.

    S: stock price
    K: strike price
    T: time to expiry
    r: risk-free rate
    sigma: volatility
    returns: delta
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0

def bs_put_price(S, K, T, r, sigma):
    """
    Black-Scholes formula for Europena put option price.
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

