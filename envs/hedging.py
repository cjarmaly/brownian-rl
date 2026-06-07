import numpy as np
import gymnasium as gym
from agents.delta_hedge import bs_put_delta, bs_put_price
from processes.gbm import simulate_gbm

"""
We've sold a put option and need to hedge it.

Each day, we need to decide how many shares to buy or sell to hedge the option.

Our P&L at each step is the gain/loss in the shares minus the gain/loss in the put option.
"""

class HedgingEnv(gym.Env):
    def __init__(self, S0, K, T, r, sigma, n_steps):
        super().__init__()
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps

        # set action space - continuous hedge ratio in [-1, 0]
        self.action_space = gym.spaces.Box(low=-1, high=0, shape=(1,), dtype=np.float32)
        
        # observation_space: [S, time_remaining, current_delta]
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, -1]), high=np.array([np.inf, T, 0]), shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.path = simulate_gbm(self.S0, self.r, self.sigma, self.T, self.n_steps, 1)[0]
        self.current_step = 0
        self.hedge_ratio = 0.0
        delta = bs_put_delta(self.S0, self.K, self.T, self.r, self.sigma)

        observation = np.array([self.S0, self.T, delta], dtype=np.float32)
        info = {}

        return observation, info

    def step(self, action):
        self.hedge_ratio = action[0]

        S_prev = self.path[self.current_step]
        self.current_step += 1
        S_curr = self.path[self.current_step]

        T_remaining = self.T - self.current_step * self.dt

        # Compute P&L
        stock_pnl = (S_curr - S_prev) * self.hedge_ratio
        put_pnl = bs_put_price(S_curr, self.K, T_remaining, self.r, self.sigma) - bs_put_price(S_prev, self.K, T_remaining + self.dt, self.r, self.sigma)
        reward = stock_pnl - put_pnl

        # compute next delta
        delta = 0 if T_remaining == 0 else bs_put_delta(S_curr, self.K, T_remaining, self.r, self.sigma)

        observation = np.array([S_curr, T_remaining, delta], dtype=np.float32)
        info = {}

        terminated = self.current_step == self.n_steps

        return observation, reward, terminated, False, info






