from stable_baselines3 import PPO
import numpy as np

"""
Train a PPO agent to hedge a put option. Use PPO to train on a continuous action space.

Proximal Policy Optimization learns a policy directly from the environment, modelling the 
probability of taking an action given the current state.
"""

def train_ppo(env, total_timesteps=500000):
    model = PPO("MlpPolicy", 
    env=env, 
    learning_rate=1e-3,
    gamma=0.99,
    verbose=1)

    model.learn(total_timesteps=total_timesteps)
    return model


"""
Evaluate the PPO hedging agent against the Black-Scholes delta hedge.
"""
def evaluate_hedging(model, env, n_episodes=1000):
    pnls = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_pnl = 0
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
            episode_pnl += reward
        pnls.append(episode_pnl)
    return np.array(pnls)