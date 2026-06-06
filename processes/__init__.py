import numpy as np

def simulate_brownian(T, n_steps, n_paths):
    """
    T: time horizon
    n_steps : number of time steps
    n_paths: number of independent paths
    returns: (n_paths, n_steps + 1) array of W_t values

    Vectorized implementation of the Brownian motion simulation.
    """

    dt = T / n_steps
    increments = np.sqrt(dt) * np.random.normal(0, 1, (n_paths, n_steps))
    paths = np.cumsum(increments, axis=1)
    W_t = np.concatenate((np.zeros((n_paths, 1)), paths), axis=1)
    return W_t