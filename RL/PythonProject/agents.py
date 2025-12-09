from stable_baselines3 import PPO, SAC, DDPG, TD3

def init_agents(envs):
    return {
        "ppo":  PPO("MlpPolicy", envs["ppo"], verbose=1),
        "sac":  SAC("MlpPolicy", envs["sac"], verbose=1),
        "ddpg": DDPG("MlpPolicy", envs["ddpg"], verbose=1),
        "td3":  TD3("MlpPolicy", envs["td3"], verbose=1),
    }

def init_agents_(envs):
    return {
        "ppo":  PPO.load("/home/hail/PycharmProjects/PythonProject/models/korea/ppo.zip", env=envs["ppo"]),
        "sac":  SAC.load("/home/hail/PycharmProjects/PythonProject/models/korea/sac.zip", env=envs["sac"]),
        "ddpg": DDPG.load("/home/hail/PycharmProjects/PythonProject/models/korea/ddpg.zip", env=envs["ddpg"]),
        "td3":  TD3.load("/home/hail/PycharmProjects/PythonProject/models/korea/td3.zip", env=envs["td3"]),
    }
