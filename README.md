# Brownian Motion, Martingales, and Reinforcement Learning

This project started as an attempt to understand something I kept encountering at the edges of my coursework: the idea that stochastic calculus and reinforcement learning are, in some deep sense, talking about the same thing. 

I want to be honest about where I started. To say I had a working knowledge of GBM and martingales is an overstatement— my knowledge was no more than several semi-successful Clause prompts asking to explain Chapter 5 of Xinfeng Zhou's "A Practical Guide to Quanititaive Finance Interviews" in **even more** detail. I'd implemented basic Q-learning agents in college coursework previously, but I'd never thought carefully about *why* importance sampling works in off-policy RL, or what it has to do with changing probability measures. This project is my attempt to close that gap.

---

## What I Built

The project is organized into five modules.

1. Implementation of Brownian motion and Geometric Brownian Motion using NumPy. Watching the paths fan out over time, always positive, never reverting — that's what lognormal actually looks like. I then implemented the Ornstein-Uhlenbeck process, which required a time-step loop because it has no clean closed form. I wish I could promise that my experiments became sharper and sharpter, but the contrast between the two plots — GBM wandering off to infinity versus OU oscillating around its long-run mean — is the clearest visual in the entire project.

2. From there I built a Gymnasium environment for American option pricing and trained a DQN agent to solve the optimal stopping problem. The agent's job is simple: at each time step, decide whether to exercise the put or keep holding. The DQN recovered 93.3% of the Longstaff-Schwartz benchmark price ($5.62 vs $6.02) without ever being told the structure of the problem. It learned the stopping rule purely from experience. Unfortunately, this did not feel like the 'win' it likely should have. I'd prefer to do better than the LS benchmark, but I guess you can't win them all.

3. The hedging module— a continuous-action Gymnasium environment where a PPO agent learns how many shares to hold to neutralize option risk. The benchmark is Black-Scholes delta hedging. After 500k training timesteps, the PPO agent had a mean P&L of -0.37 compared to Black-Scholes's -2.15, but a standard deviation of 9.08 versus 1.24. The RL agent's higher mean is flattering but misleading— what matters for a hedger is variance, and by that measure Black-Scholes wins by a factor of seven. Ouch.

4. The Girsanov module. The central claim is that you can price a European put correctly by simulating paths under the real-world measure (drift $\mu$) and reweighting each path's payoff by the Radon-Nikodym derivative $L_T$. I verified this empirically: pricing under $\mathbb{P}$ gave $3.96$, reweighting by $L_T$ gave $5.59$, and Black-Scholes gave $5.57$. A 0.3% gap at 100,000 paths shows quite beautifully a mathematically dense derivation holding in practice. A win for me, albeit an expected one.

5. The final module computed VaR and CVaR over the hedging P&L distributions. The key number is the CVaR/VaR ratio: 1.05 for Black-Scholes, 1.31 for the RL agent. A ratio near 1.0 means the tail is thin and well-characterized. A ratio of 1.31 means that once you breach the VaR threshold, things get meaningfully worse than VaR implies. This is exactly why modern risk frameworks (Basel III) mandate CVaR over VaR — and it's a concrete, empirical illustration of why an under-trained RL agent is not yet safe to deploy.

---

## What I Found

A few results stood out.

The DQN's Q-function, at convergence, approximates the Snell envelope— the smallest supermartingale dominating the payoff function. I didn't set out to verify this formally, but the numerical agreement with Longstaff-Schwartz is consistent with it. The agent learned something tangible about the structure of optimal stopping without being told what that structure was. For someone still new to deep learniing (me), this still brings about a certain astonishment.

The Girsanov verification confirmed that $\mathbb{E}[L_T] = 1.0005$ at 100,000 paths, and that the reweighted mean of $W_T$ under $\mathbb{Q}$ (-0.2523) matches the exact analytical value (-0.2500) to within 1%. These are not interesting numbers on their own, but they matter because they confirm the implementation is correct before using it to price anything.

The most practically interesting result is probably the CVaR gap. An RL agent that looks competitive on mean P&L can have tail risk nearly five times worse than the benchmark. That's the kind of thing that gets missed if you only look at average performance, highlighting the danger of overfitting models to a single performance metric.

---

## The Connection I Was Looking For

Honestly, I built the Girsanov module primarily to understand the Radon-Nikodym derivative, and not because I had a grand unified theory waiting to be verified.The connection to RL was something I read about and was curious to pursue. But it does hold up. 

You have data collected under one probability measure (the behavior policy, or the real-world drift $\mu$), and you want to evaluate something under a different one (the target policy, or the risk-neutral drift $r$). The correction in both cases is a likelihood ratio that reweights each outcome for the mismatch. In continuous time it's the Girsanov exponential martingale. In discrete time it's $\prod_t \pi(a_t|s_t) / \beta(a_t|s_t)$. What a beautiful commonality.

The somewhat deflating punchline is that neither of our agents actually computes this correction. DQN leans on a replay buffer to smooth over distribution shift without correcting for it. PPO clips updates to bound the shift, which is more principled but still an approximation. A fully rigorous implementation would multiply each trajectory's return by $L_T$​ before updating. That's where I ran out of runway.

---

## What I'd Do Next
Three things, in order of how much they irk me.

1. The PPO agent optimizes for mean P&L, which is the wrong objective for a hedger. Variance is what kills you! Adding $-\lambda \cdot |\text{P\&L}|$ to the reward is a straightforward fix I didn't get to. I suspect it would meaningfully close the CVaR gap.

2. The replay buffer in DQN is doing a lot of work it shouldn't have to do. Computing LT​ for each stored trajectory and reweighting explicitly would be a cleaner implementation of what the buffer approximates crudely. Whether it actually helps convergence is an open question.

3. Finally, the whole project is built under Black-Scholes assumptions. Not only does this hand the analytical benchmark an unfair advantage, if there is one thing I've learned from Dan Rasmussen, it is to be weary of utopian frameworks. No transaction costs, constant volatility, daily rebalancing. Add costs and the RL agent's flexibility might start to tip the scales the other direction.

---

## A Note

This is far from a research paper. The RL training is short, the sample sizes are modest, and I've left more questions open than I've answered. The numbers are what they are (I did not cherry-pick runs), and where the agents underperform, I tried to say so clearly rather than reach for an excuse.

Here's to trying and doing mediocre, but learning. 


## Project Structure

```
brownian-rl/
├── processes/
│   ├── brownian.py       # Brownian motion simulation
│   ├── gbm.py            # Geometric Brownian Motion
│   └── ou.py             # Ornstein-Uhlenbeck process
├── envs/
│   ├── american_option.py  # Gymnasium env for optimal stopping
│   └── hedging_env.py      # Gymnasium env for delta hedging
├── agents/
│   ├── dqn.py              # DQN training
│   ├── ppo.py              # PPO training and evaluation
│   ├── longstaff_schwartz.py  # LSM benchmark
│   ├── delta_hedge.py      # Black-Scholes delta and price
│   └── girsanov.py         # Radon-Nikodym derivative
├── risk/
│   └── metrics.py          # VaR and CVaR
└── experiments/
    ├── gbm_paths/           # Experiment 01
    ├── ou_paths/            # Experiment 02
    ├── dqn_american_put/    # Experiment 03
    ├── dqn_vs_lsm/          # Experiment 04
    ├── ppo_hedging_vs_bs/   # Experiment 05
    ├── girsanov_pricing/    # Experiment 06
    └── hedging_tail_risk/   # Experiment 07
```

## Setup

```bash
git clone https://github.com/your-username/brownian-rl
cd brownian-rl
python -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib jupyter torch stable-baselines3 gymnasium
```

Run any experiment with:

```bash
python -m experiments.<experiment_name>.run
```