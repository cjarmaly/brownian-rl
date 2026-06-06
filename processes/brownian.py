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

if __name__ == "__main__":
    W = simulate_brownian(T=1, n_steps=100, n_paths=1000)

    # Check all paths start at 0
    print(W[:, 0])

    # Check at t=0.5 (step 50), variance should be ~0.5
    print(np.var(W[:, 50]))

    # Check at t=1 (step 1000), variance should be ~1
    print(np.var(W[:, 100]))