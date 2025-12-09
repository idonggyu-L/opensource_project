import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DDPG, TD3
from gymnasium.envs.registration import register
import pandas as pd
import numpy as np
from gymnasium import spaces

class StocksEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path, window_size=10, eps_length=200,
                 vol_window=30, beta=0.05, trade_cost=0.001, hold_cost=0.0,
                 initial_cash=10000, reward_scale=0.01):
        super().__init__()
        self.csv_path = csv_path
        self.window_size = window_size
        self.eps_length = eps_length
        self.vol_window = vol_window
        self.beta = beta
        self.trade_cost = trade_cost
        self.hold_cost = hold_cost
        self.initial_cash = initial_cash
        self.reward_scale = reward_scale

        # load and normalize
        raw_data = pd.read_csv(csv_path)
        self.raw_prices = raw_data["Close"].to_numpy(dtype=np.float64)
        self.raw_prices = np.clip(self.raw_prices, 1e-6, None)

        norm_prices = (self.raw_prices - self.raw_prices.mean()) / (self.raw_prices.std() + 1e-8)
        diff = np.insert(np.diff(norm_prices), 0, 0)
        volume = raw_data["Volume"].to_numpy(dtype=np.float64)
        self.features = np.column_stack([norm_prices, diff, volume])

        # log returns
        logp = np.log(np.clip(self.raw_prices, 1e-6, None))
        self.logret = np.zeros_like(logp)
        self.logret[1:] = logp[1:] - logp[:-1]

        # rolling sigma
        self.roll_sigma = np.full_like(self.logret, 1e-6)
        for t in range(vol_window + 1, len(self.logret)):
            w = self.logret[t - vol_window + 1: t + 1]
            self.roll_sigma[t] = max(np.std(w), 1e-6)

        feat_dim = self.features.shape[1]
        obs_dim = window_size * feat_dim + 2

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.start_idx = self.np_random.integers(self.window_size, len(self.raw_prices) - self.eps_length)
        self.current_tick = self.start_idx
        self.end_tick = self.start_idx + self.eps_length
        self.position = 0.0
        self.last_trade_tick = self.current_tick - 1
        self.total_reward = 0.0

        # capital tracking
        self.cash = self.initial_cash
        self.equity = self.initial_cash
        self.equity_curve = [self.initial_cash]

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = float(action[0]) if isinstance(action, (np.ndarray, list)) else float(action)
        self.current_tick += 1
        terminated = self.current_tick >= self.end_tick
        truncated = False

        prev_position = self.position
        self.position = np.clip(action, -1.0, 1.0)

        reward = self._calculate_reward(prev_position, self.position)
        if not np.isfinite(reward):
            reward = 0.0
        self.total_reward += reward

        prev_price = self.raw_prices[self.current_tick - 1]
        curr_price = self.raw_prices[self.current_tick]
        price_change = curr_price - prev_price
        self.equity += self.position * price_change
        if not np.isfinite(self.equity):
            self.equity = self.initial_cash
        self.equity_curve.append(self.equity)

        obs = self._get_obs()
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs)

        info = {
            "total_reward": self.total_reward,
            "position": self.position,
            "equity": self.equity,
            "equity_curve": self.equity_curve
        }
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, prev_position, new_position):
        t = self.current_tick
        logret = self.logret[t]

        z = (new_position * logret) / max(self.roll_sigma[t], 1e-6)
        r_core = np.tanh(self.beta * z)

        trade_cost = self.trade_cost * abs(new_position - prev_position)
        hold_cost = self.hold_cost * abs(new_position)

        reward = float(r_core - trade_cost - hold_cost)
        reward *= self.reward_scale  # scale reward down
        return reward

    def _get_obs(self):
        start = self.current_tick - self.window_size + 1
        frame = self.features[start:self.current_tick + 1].reshape(-1)
        tick = (self.current_tick - self.last_trade_tick) / self.eps_length
        obs = np.hstack([frame, [self.position], [tick]])
        return obs.astype(np.float32)

    def render(self):
        pass
def run_episode(env, model):
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["equity_curve"]

if __name__ == "__main__":
    register(
        id="StocksContinuous-v0",
        entry_point="__main__:StocksEnv",
        kwargs={"csv_path": "/home/hail/PycharmProjects/data/102110.KS.csv",
                "initial_cash": 1000000},
    )

    env = gym.make("StocksContinuous-v0")

    ppo = PPO.load("/home/hail/PycharmProjects/PythonProject/ppo_stock_model", env=env)
    sac = SAC.load("sac_stock_model", env=env)
    ddpg = DDPG.load("ddpg_stock_model", env=env)
    td3 = TD3.load("td3_stock_model", env=env)

    # Å×½ºÆ® ½ÇÇà
    equity_ppo = run_episode(env, ppo)
    equity_sac = run_episode(env, sac)
    equity_ddpg = run_episode(env, ddpg)
    equity_td3 = run_episode(env, td3)

    # ½Ã°¢È­
    plt.figure(figsize=(10, 6))
    plt.plot(equity_ppo, label="PPO")
    plt.plot(equity_sac, label="SAC")
    plt.plot(equity_ddpg, label="DDPG")
    plt.plot(equity_td3, label="TD3")
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.title("Equity Curve Comparison (New CSV)")
    plt.legend()
    plt.grid(True)
    plt.show()
