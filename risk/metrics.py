import numpy as np

"""
Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) for a given portfolio of P&L values.

VaR is the maximum loss at the given confidence level.
CVaR is the average loss within beyond the VaR.
"""

def var(pnls, alpha=0.95):
    """
    Calculate Value at Risk at confidence level alpha.
    pnls  : array of P&L values
    alpha : confidence level
    returns: VaR (positive number = loss)
    """
    return -np.percentile(pnls, 100 * (1 - alpha)) # negative because we want to return a loss

def cvar(pnls, alpha=0.95):
    """
    Calculate Conditional Value at Risk at confidence level alpha.
    pnls  : array of P&L values
    alpha : confidence level
    returns: CVaR (positive number = loss)
    """

    #compute VaR then average the losses beyond it
    var_threshold = var(pnls, alpha)
    return -np.mean(pnls[pnls <= -var_threshold])