import numpy as np

def longstaff_schwartz(paths, K, r, dt):
    """
    Longstaff-Schwartz algorithm for American option pricing.

    paths: (n_paths, n_steps + 1) array of stock pricepaths
    K: strike price
    r: risk-free rate
    dt: time step size
    returns: (n_paths, n_steps + 1) array of option values
    """
    n_paths, n_steps_1 = paths.shape
    n_steps = n_steps_1 - 1


    # Compute payoff at expiry
    payoffs = np.maximum(0, K - paths[:, -1])

    # Iterate backwards
    for t in range(n_steps - 1, 0, -1):

        # discount payoffs one time step
        payoffs = payoffs * np.exp(-r * dt)

        # get paths that are in the money
        itm = paths[:, t] < K

        # run regression on discounted payoffs of in-the-money paths on basis functions
        if np.sum(itm) > 0:
            S = paths[:, t][itm] # stock prices of in-the-money paths at time t
            Y = payoffs[itm] # discounted payoffs of in-the-money paths at time t

            # fit linear regression model
            X = np.column_stack([
                np.ones_like(S),
                S,
                S**2,
            ])

            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

            # compute continuation value (fitted regression values)
            continuation = X @ coeffs

            # check if exercising now is optimal
            intrinsic = K - S
            exercise_now = intrinsic > continuation 

            # update payoffs for in-the-money paths where exercising now is optimal
            itm_indices = np.where(itm)[0]
            payoffs[itm_indices[exercise_now]] = intrinsic[exercise_now]

        return np.mean(payoffs) * np.exp(r * dt) # discount back to present value





