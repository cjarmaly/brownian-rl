import gymnasium as gym
import numpy as np
from processes.gbm import simulate_gbm

class AmericanOption(gym.Env):
    def __init__(self, S0, K, T, r, sigma, n_steps):
        """
        S0: initial stock price
        K: strike price
        T: time horizon
        r: risk-free rate
        sigma: volatility
        n_steps: number of time steps
        """
        super().__init__() # initialize parent class gym.Env
        # store parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r # we store r instead of mu under risk-neutral Black-Scholes assumptions
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps

        # define action_space (continue, exercise)
        self.action_space = gym.spaces.Discrete(2)

        # define observation_space (stock price, time to maturity)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # simulate stock price (GBM) path
        self.path = simulate_gbm(self.S0, self.r, self.sigma, self.T, self.n_steps, 1)

        # set the current step to 0
        self.current_step = 0

        # initial observation is the stock price at the first step
        observation = np.array([self.path[0][0], self.T], dtype=np.float32)
        info = {} # dictionary for debugging and logging

        return observation, info

    def step(self, action):
        # get current price
        current_price = self.path[0][self.current_step]
        if action == 1 or self.current_step == self.n_steps: # exercise option at expiry
            reward = max(0, self.K - current_price)
            terminated = True
        else: # continue holding
            reward = 0
            self.current_step += 1
            terminated = False

        observation = np.array([self.path[0][self.current_step], self.T - self.current_step * self.dt], dtype=np.float32)
        info = {}

        return observation, reward, terminated, False, info 