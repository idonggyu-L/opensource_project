from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DDPG, TD3
from gymnasium.envs.registration import register


class StocksEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path, window_size=10, eps_length=100,
                 vol_window=30, beta=0.05, trade_cost=0.00, hold_cost=0.0,
                 initial_cash=1000000, reward_scale=0.01):
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

        # ====== 데이터 로드 ======
        raw_data = pd.read_csv(csv_path)
        self.raw_prices = raw_data["Close"].to_numpy(dtype=np.float64)
        self.raw_prices = np.clip(self.raw_prices, 1e-6, None)
        volume = raw_data["Volume"].to_numpy(dtype=np.float64)

        # ====== 지표 계산 ======
        df = raw_data.copy()

        # EMA (지수 이동 평균, 기간 14로 예시)
        df["EMA"] = df["Close"].ewm(span=14, adjust=False).mean()

        # RSI (상대강도지수, 기간 14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df["RSI"] = 100 - (100 / (1.0 + rs))

        # ATR (Average True Range, 기간 14)
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14, min_periods=14).mean()

        # OBV (On Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv

        # ====== feature 구성 ======
        norm_prices = (self.raw_prices - self.raw_prices.mean()) / (self.raw_prices.std() + 1e-8)
        diff = np.insert(np.diff(norm_prices), 0, 0)

        ema = np.nan_to_num(df["EMA"].to_numpy(dtype=np.float64), nan=0.0)
        rsi = np.nan_to_num(df["RSI"].to_numpy(dtype=np.float64), nan=50.0)
        atr = np.nan_to_num(df["ATR"].to_numpy(dtype=np.float64), nan=0.0)
        obv = np.nan_to_num(df["OBV"].to_numpy(dtype=np.float64), nan=0.0)

        self.features = np.column_stack([norm_prices, diff, volume, ema, rsi, atr, obv])

        # ====== 로그 수익률 ======
        logp = np.log(np.clip(self.raw_prices, 1e-6, None))
        self.logret = np.zeros_like(logp)
        self.logret[1:] = logp[1:] - logp[:-1]

        # ====== 롤링 표준편차 ======
        self.roll_sigma = np.full_like(self.logret, 1e-6)
        for t in range(vol_window + 1, len(self.logret)):
            w = self.logret[t - vol_window + 1: t + 1]
            self.roll_sigma[t] = max(np.std(w), 1e-6)

        # ====== 관측/행동 공간 정의 ======
        feat_dim = self.features.shape[1]  # 이제 7
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
        reward *= self.reward_scale
        return reward

    def _get_obs(self):
        start = self.current_tick - self.window_size + 1
        frame = self.features[start:self.current_tick + 1].reshape(-1)
        tick = (self.current_tick - self.last_trade_tick) / self.eps_length
        obs = np.hstack([frame, [self.position], [tick]])
        return obs.astype(np.float32)

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

if __name__ == "__main__":
    register(
        id="StocksContinuous-v0",
        entry_point=__name__ + ":StocksEnv",
        kwargs={"csv_path": "/home/hail/PycharmProjects/PythonProject/env/Korea/069500.KS.csv",
                "initial_cash": 1000000},
    )

    # Random
    # env = gym.make("StocksContinuous-v0")
    # equity_random = run_episode(env, random=True)

    # PPO
    env = gym.make("StocksContinuous-v0")
    ppo = PPO("MlpPolicy", env, verbose=1)
    ppo.learn(total_timesteps=100000)
    # equity_ppo = run_episode(env, model=ppo)
    #
    # # SAC
    # env = gym.make("StocksContinuous-v0")
    # sac = SAC("MlpPolicy", env, verbose=1,  learning_rate=1e-4)
    # sac.learn(total_timesteps=10000)
    # equity_sac = run_episode(env, model=sac)
    #
    # # DDPG
    # env = gym.make("StocksContinuous-v0")
    # ddpg = DDPG("MlpPolicy", env, verbose=1)
    # ddpg.learn(total_timesteps=10000)
    # equity_ddpg = run_episode(env, model=ddpg)
    #
    # #TD3
    # env = gym.make("StocksContinuous-v0")
    # td3 = TD3("MlpPolicy", env, verbose=1)
    # td3.learn(total_timesteps=10000)
    # equity_td3 = run_episode(env, model=td3)

    ppo.save("ppo_stock_")
    # sac.save("sac_stock_model")
    # ddpg.save("ddpg_stock_model")
    # td3.save("td3_stock_model")
    #
    # #Plot comparison
    # plt.figure(figsize=(10, 6))
    # plt.plot(equity_random, label="Random", linestyle="--")
    # plt.plot(equity_ppo, label="PPO")
    # plt.plot(equity_sac, label="SAC")
    # plt.plot(equity_ddpg, label="DDPG")
    # plt.plot(equity_td3, label="TD3")
    # plt.xlabel("Steps")
    # plt.ylabel("Equity")
    # plt.title("Equity Curve Comparison (Random vs PPO vs SAC vs DDPG vs TD3)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
