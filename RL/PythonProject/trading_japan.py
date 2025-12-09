import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO, DDPG, A2C, SAC
from stable_baselines3.common.callbacks import EvalCallback

from gymnasium import spaces
from gymnasium.envs.registration import register


class StocksEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path, window_size=30, eps_length=200,
                 trade_cost=0.001, hold_cost=0.0,
                 initial_cash=1_000_000, reward_scale=1.0):
        super().__init__()
        self.csv_path = csv_path
        self.window_size = window_size
        self.eps_length = eps_length
        self.trade_cost = trade_cost
        self.hold_cost = hold_cost
        self.initial_cash = initial_cash
        self.reward_scale = reward_scale

        df = pd.read_csv(csv_path)
        self.raw_prices = df["Close"].to_numpy(dtype=np.float64)
        self.raw_prices = np.clip(self.raw_prices, 1e-6, None)

        close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

        # ===== Technical Indicators =====
        df["EMA"] = close.ewm(span=14, adjust=False).mean()

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df["RSI"] = 100 - (100 / (1.0 + rs))

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()

        obv = [0]
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        df["MACD"], df["MACD_signal"] = macd, signal

        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["Bollinger_upper"] = ma20 + 2 * std20
        df["Bollinger_lower"] = ma20 - 2 * std20

        df["Momentum"] = close.diff(10)
        df["Volatility20"] = close.pct_change().rolling(20).std()

        # ===== Feature Matrix =====
        norm_prices = (self.raw_prices - self.raw_prices.mean()) / (self.raw_prices.std() + 1e-8)
        diff = np.insert(np.diff(norm_prices), 0, 0)

        features = np.column_stack([
            norm_prices, diff, volume,
            df["EMA"], df["RSI"], df["ATR"], df["OBV"],
            df["MACD"], df["MACD_signal"],
            df["Bollinger_upper"], df["Bollinger_lower"],
            df["Momentum"], df["Volatility20"]
        ])

        # normalize features (z-score) + NaN safe
        self.features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=1e6, neginf=-1e6)

        # log returns
        logp = np.log(np.clip(self.raw_prices, 1e-6, None))
        self.logret = np.zeros_like(logp)
        self.logret[1:] = logp[1:] - logp[:-1]

        feat_dim = self.features.shape[1]
        obs_dim = window_size * feat_dim + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # action space: Discrete(3) = Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.start_idx = self.np_random.integers(self.window_size, len(self.raw_prices) - self.eps_length)
        self.current_tick = self.start_idx
        self.end_tick = self.start_idx + self.eps_length
        self.position = 0.0
        self.last_trade_tick = self.current_tick - 1
        self.total_reward = 0.0

        self.cash = self.initial_cash
        self.equity = self.initial_cash
        self.equity_curve = [self.initial_cash]

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # map action {0: Sell, 1: Hold, 2: Buy}
        if action == 0:
            new_position = -1.0
        elif action == 1:
            new_position = 0.0
        else:
            new_position = 1.0

        self.current_tick += 1
        terminated = self.current_tick >= self.end_tick
        truncated = False

        prev_position = self.position
        self.position = new_position

        # reward = position * price change
        prev_price = self.raw_prices[self.current_tick - 1]
        curr_price = self.raw_prices[self.current_tick]
        price_change = (curr_price - prev_price) / prev_price
        reward = (self.position * price_change) - (self.trade_cost * abs(new_position - prev_position))
        reward *= self.reward_scale
        if not np.isfinite(reward):
            reward = 0.0
        self.total_reward += reward
        self.equity += self.position * (curr_price - prev_price)
        if not np.isfinite(self.equity):
            self.equity = self.initial_cash
        self.equity_curve.append(self.equity)

        obs = self._get_obs()
        info = {
            "total_reward": self.total_reward,
            "position": self.position,
            "equity": self.equity,
            "equity_curve": self.equity_curve
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        start = self.current_tick - self.window_size + 1
        frame = self.features[start:self.current_tick + 1].reshape(-1)
        tick = (self.current_tick - self.last_trade_tick) / self.eps_length
        obs = np.hstack([frame, [self.position], [tick]])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs

    def render(self):
        pass


def run_episode(env, model=None, random=False):
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        if random:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["equity_curve"]


def train_and_eval(model_class, name, **kwargs):
    env = gym.make("StocksDiscrete-v0")
    eval_env = StocksEnv(eval_path, window_size=30, eps_length=200)
    eval_callback = EvalCallback(eval_env, eval_freq=100, n_eval_episodes=5, verbose=1)
    model = model_class("MlpPolicy", env, verbose=1, **kwargs)
    model.learn(total_timesteps=100000, callback=eval_callback)
    equity_curve = run_episode(eval_env, model=model)

    df_result = pd.DataFrame({
        "algo": name,
        "timesteps": eval_callback.evaluations_timesteps,
        "mean_reward": [np.mean(r) for r in eval_callback.evaluations_results],
        "mean_length": [np.mean(l) for l in eval_callback.evaluations_length]
    })
    all_eval_results.append(df_result)

    save_dir = "./models/japan"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, name))

    return equity_curve


if __name__ == "__main__":
    csv_path = "/home/hail/Desktop/stock/ETF/Japan/1321.T.csv"
    df = pd.read_csv(csv_path)

    split_idx = int(len(df) * 0.8)
    train_path = "./env/jp/train.csv"
    eval_path = "./env/jp/eval.csv"
    df.iloc[:split_idx].to_csv(train_path, index=False)
    df.iloc[split_idx:].to_csv(eval_path, index=False)

    register(
        id="StocksDiscrete-v0",
        entry_point=__name__ + ":StocksEnv",
        kwargs={"csv_path": train_path,
                "initial_cash": 1_000_000},
    )

    # Random baseline
    env = gym.make("StocksDiscrete-v0")
    equity_random = run_episode(env, random=True)

    all_eval_results = []

    equity_ppo = train_and_eval(PPO, "ppo")
    equity_a2c = train_and_eval(A2C, "a2c")
    # equity_ddpg = train_and_eval(SAC, "sac")

    eval_results = pd.concat(all_eval_results, ignore_index=True)
    eval_results.to_csv("j_eval_results.csv", index=False)

    # === Plot comparison ===
    plt.figure(figsize=(10, 6))
    plt.plot(equity_random, label="Random", linestyle="--")
    plt.plot(equity_ppo, label="PPO")
    plt.plot(equity_a2c, label="A2C")
    # plt.plot(equity_ddpg, label="DDPG")
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.title("Equity Curve Comparison (Random vs PPO vs A2C vs DDPG)")
    plt.legend()
    plt.grid(True)
    plt.show()
