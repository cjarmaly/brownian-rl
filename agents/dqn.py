from stable_baselines3 import DQN

"""
Deep Q-Network learns a Q-function that maps states to expected future rewards given actions.
Fits a discrete action space.
"""

def train_dqn(env, total_timesteps=100000):
    model = DQN(
        policy="MlpPolicy", 
        env=env, 
        learning_rate=1e-3, 
        buffer_size=10000, 
        batch_size=64, 
        gamma=0.99, 
        verbose=1
        )
    model.learn(total_timesteps=total_timesteps)
    return model