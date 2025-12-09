import os
import pandas as pd
import torch
import gymnasium as gym
from gymnasium.envs.registration import register

from env import StocksEnv
from agents import init_agents_
from arbitrator import MetaControllerHardSelect
from arbitrator_ import MetaControllerSoftSelect


if __name__ == "__main__":
    # === 데이터 준비 ===
    csv_path = "/home/hail/Desktop/stock/ETF/Korea/069500.KS.csv"
    df = pd.read_csv(csv_path)
    split_idx = int(len(df) * 0.8)
    train_path = "./train.csv"
    eval_path  = "./eval.csv"
    df.iloc[:split_idx].to_csv(train_path, index=False)
    df.iloc[split_idx:].to_csv(eval_path, index=False)

    # === 환경 등록 ===
    try:
        register(
            id="StocksContinuous-v0",
            entry_point="env:StocksEnv",
            kwargs={"csv_path": train_path, "initial_cash": 1_000_000},
        )
    except Exception:
        pass

    env_main = gym.make("StocksContinuous-v0")
    train_envs = {
        "ppo":  gym.make("StocksContinuous-v0"),
        "sac":  gym.make("StocksContinuous-v0"),
        "ddpg": gym.make("StocksContinuous-v0"),
        "td3":  gym.make("StocksContinuous-v0"),
    }

    # === 에이전트 초기화 ===
    agents = init_agents_(train_envs)

    # === 메타 컨트롤러 ===
    meta = MetaControllerSoftSelect(
        env_main=env_main,
        train_envs=train_envs,
        agents=agents,
        gamma=1.0,
        meta_lr=3e-4,
        device="cpu",
    )

    # === 학습 실행 ===
    for ep in range(50000):
        ret = meta.run_episode(max_steps=env_main.unwrapped.eps_length)
        print(f"[Episode {ep+1}] Return={ret:.4f}")

    # === 모델 저장 ===
    os.makedirs("models", exist_ok=True)

    # SB3 하위 에이전트 저장
    agents["ppo"].save("models/ppo_agent")
    agents["sac"].save("models/sac_agent")
    agents["ddpg"].save("models/ddpg_agent")
    agents["td3"].save("models/td3_agent")

    # 메타 정책 저장 (PyTorch)
    torch.save({
        "meta_pi": meta.meta_pi.state_dict(),
        "meta_v": meta.meta_v.state_dict(),
        "opt": meta.opt.state_dict()
    }, "models/meta_policy.pt")

    print("✅ 모든 모델 저장 완료!")

    # === 모델 불러오기 예시 ===
    # from stable_baselines3 import PPO, SAC, DDPG, TD3
    # ppo_loaded = PPO.load("models/ppo_agent")
    # checkpoint = torch.load("models/meta_policy.pt", map_location="cpu")
    # meta.meta_pi.load_state_dict(checkpoint["meta_pi"])
    # meta.meta_v.load_state_dict(checkpoint["meta_v"])
    # meta.opt.load_state_dict(checkpoint["opt"])
    # print("✅ 모델 불러오기 완료!")
