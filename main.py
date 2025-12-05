from retrieval_news import search_news
from retrieval_terms import explain_terms
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from ppo import m as model


##API key##

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_input: str

class StockRequest(BaseModel):
    ticker_code: str

llm = ChatOpenAI(model="gpt-4o-mini")

# 용어 추출 체인
keyword_prompt = PromptTemplate(
    template="""
    아래 뉴스 요약에서 경제/금융 관련 핵심 용어 3~5개만 뽑아서
    반드시 JSON 배열 형식으로만 출력해라.
    예시: ["기준금리","FOMC","금리"]

    뉴스 요약:
    {context}
    """,
    input_variables=["context"]
)
keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt, verbose=True)

def extract_terms_with_llm(context: str) -> list[str]:
    raw_output = keyword_chain.run(context=context)
    try:
        terms = json.loads(raw_output)
        if isinstance(terms, list):
            return [t.strip() for t in terms]
    except Exception as e:
        print("JSON parse 실패:", e, "| raw_output:", raw_output)
    return []

def route_query(user_input: str):
    # 1) 뉴스 검색 + 요약
    news_result = search_news(user_input, k=3)
    news_summary = news_result["summary"]
    news_links = news_result["links"]

    # 2) 뉴스 요약 기반 용어 추출
    keywords = extract_terms_with_llm(news_summary)
    print(keywords)

    # 3) 용어 설명 (사전 DB 참고)
    terms_answer = {}
    for term in keywords:
        explanation = explain_terms(term)
        print(explanation)
        if explanation and "찾지 못했습니다" not in explanation:
            terms_answer[term] = explanation
    
    if "관련 뉴스를 찾지 못했습니다" in news_summary:
        return {
            "뉴스 요약": news_summary,
            "관련 용어": terms_answer,
            "출처": []   
        }

    # 4) 최종 응답 구조
    response = {
        "뉴스 요약": news_summary,
        "관련 용어": terms_answer,
        "출처": news_links
    }
    print(response)
    return response

@app.post("/query")
def query_api(request: QueryRequest):
    print("/query called with:", request.user_input)  # 요청 확인

    try:
        resp = route_query(request.user_input)
        print("/query response:", resp)  # 성공 시 응답 확인
        return resp

    except Exception as e:
        print("/query error:", repr(e))
        return {"error": str(e)}
    #return route_query(request.user_input)

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


        raw_data = pd.read_csv(csv_path)
        self.raw_prices = raw_data["Close"].to_numpy(dtype=np.float64)
        self.raw_prices = np.clip(self.raw_prices, 1e-6, None)
        volume = raw_data["Volume"].to_numpy(dtype=np.float64)

        df = raw_data.copy()
        df["EMA"] = df["Close"].ewm(span=14, adjust=False).mean()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df["RSI"] = 100 - (100 / (1.0 + rs))

        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14, min_periods=14).mean()

        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv

        norm_prices = (self.raw_prices - self.raw_prices.mean()) / (self.raw_prices.std() + 1e-8)
        diff = np.insert(np.diff(norm_prices), 0, 0)

        ema = np.nan_to_num(df["EMA"].to_numpy(dtype=np.float64), nan=0.0)
        rsi = np.nan_to_num(df["RSI"].to_numpy(dtype=np.float64), nan=50.0)
        atr = np.nan_to_num(df["ATR"].to_numpy(dtype=np.float64), nan=0.0)
        obv = np.nan_to_num(df["OBV"].to_numpy(dtype=np.float64), nan=0.0)

        self.features = np.column_stack([norm_prices, diff, volume, ema, rsi, atr, obv])

        logp = np.log(np.clip(self.raw_prices, 1e-6, None))
        self.logret = np.zeros_like(logp)
        self.logret[1:] = logp[1:] - logp[:-1]

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
        if len(self.raw_prices) <= self.eps_length:
            self.start_idx = 0
            self.end_tick = len(self.raw_prices)
        else:
            self.start_idx = self.np_random.integers(
            self.window_size, len(self.raw_prices) - self.eps_length
            )
            self.end_tick = self.start_idx + self.eps_length
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
        terminated = self.current_tick >= (self.end_tick-1)
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
        start = max(self.current_tick - self.window_size + 1, 0)
        frame = self.features[start:self.current_tick + 1]
        if frame.shape[0] < self.window_size:
            pad_len = self.window_size - frame.shape[0]
            pad = np.zeros((pad_len, frame.shape[1]))  # feat_dim ?? ??
            frame = np.vstack((pad, frame))
        frame = frame.reshape(-1)
        tick = (self.current_tick - self.last_trade_tick) / self.eps_length
        obs = np.hstack([frame, [self.position], [tick]])
        return obs.astype(np.float32)

    def render(self):
        pass

def predict_stock_action(ticker_code: str,
                         model_path: str = "./ppo_stock_.zip",
                         window_size: int = 10,
                         eps_length: int = 40,
                         period: str = "60d"):
    ticker = ticker_code.strip() + ".KS"
    #ticker = yf.Ticker(ticker)
    #data = ticker.history(interval='1d', period='60d', auto_adjust=False)

    data = yf.download(ticker, period=period, interval="1d")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.sort_index()

    data = data.tail(eps_length)
    if len(data) < eps_length:
        return {"error": f"데이터 개수가 {eps_length}보다 적습니다."}

    eval_path = "eval.csv"
    data.to_csv(eval_path)

    env = StocksEnv(eval_path, window_size=window_size, eps_length=eps_length)


    obs, _ = env.reset()
    done = False
    last_action = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_action = float(action[0])  # 마지막 액션만 저장
    if last_action > 0:
        return {"종목": ticker_code, "예측": f"매수 의견 {round(last_action * 100, 1)}%"}
    elif last_action < 0:
        return {"종목": ticker_code, "예측": f"매도 의견 {round(abs(last_action) * 100, 1)}%"}
    else:
        return {"종목": ticker_code, "예측": "중립 (보유)"}

@app.post("/predict_stock")
def predict_stock_api(request: StockRequest):
    try:
        return predict_stock_action(request.ticker_code)
    except Exception as e:
        return {"error": str(e)}


