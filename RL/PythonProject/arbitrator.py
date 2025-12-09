import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reliability import BayesRelEstimator


# ===============================
# Meta Policy & Value Networks
# ===============================
class MetaPolicy(nn.Module):
    def __init__(self, obs_dim, n_agents=4, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.logits = nn.Linear(hidden, n_agents)

    def forward(self, obs_t, chis_t):
        x = torch.cat([obs_t, chis_t], dim=-1)
        h = self.body(x)
        logits = self.logits(h)
        return logits  # (B, n_agents)


class MetaValue(nn.Module):
    def __init__(self, obs_dim, n_agents=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs_t, chis_t):
        x = torch.cat([obs_t, chis_t], dim=-1)
        return self.net(x)  # (B, 1)


# ===============================
# Meta Controller (Hard Selection)
# ===============================
class MetaControllerHardSelect:
    def __init__(self, env_main, train_envs, agents,
                 gamma=1.0, meta_lr=3e-4,
                 device="cpu"):
        self.env = env_main
        self.train_envs = train_envs
        self.agents = agents
        self.order = list(agents.keys())
        self.n_agents = len(self.order)
        self.gamma = gamma
        self.device = device

        obs_dim = env_main.observation_space.shape[0]
        self.meta_pi = MetaPolicy(obs_dim, n_agents=self.n_agents).to(device)
        self.meta_v  = MetaValue (obs_dim, n_agents=self.n_agents).to(device)
        self.opt = optim.Adam(list(self.meta_pi.parameters()) +
                              list(self.meta_v.parameters()), lr=meta_lr)

        self.rel = {k: BayesRelEstimator() for k in self.order}

        # rollout buffers
        self.buf_obs, self.buf_chis, self.buf_rew = [], [], []
        self.buf_done, self.buf_val, self.buf_act = [], [], []

    def _clear_buf(self):
        self.buf_obs.clear(); self.buf_chis.clear()
        self.buf_rew.clear(); self.buf_done.clear()
        self.buf_val.clear(); self.buf_act.clear()

    @torch.no_grad()
    def _all_actions_and_values(self, obs):
        actions, values = [], []
        for name in self.order:
            agent = self.agents[name]
            a, _ = agent.predict(obs, deterministic=True)
            actions.append(a)
            values.append(0.0)
        return actions, values

    def run_episode(self, max_steps=500):
        obs, _ = self.env.reset()
        done, step, ep_ret = False, 0, 0.0

        while not done and step < max_steps:
            actions, values = self._all_actions_and_values(obs)

            chis_np = np.array([self.rel[name].reliability() for name in self.order], dtype=np.float32)
            obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            chis_t = torch.as_tensor(chis_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            # === Hard selection ===
            logits = self.meta_pi(obs_t, chis_t)
            probs = torch.softmax(logits, dim=-1)        # (1, n_agents)
            best_idx = torch.argmax(probs, dim=-1).item()  # agent index
            a_sel = actions[best_idx]                     # agent action

            a_sel = np.clip(a_sel, self.env.action_space.low, self.env.action_space.high)
            val = self.meta_v(obs_t, chis_t).squeeze(-1)

            next_obs, r, terminated, truncated, info = self.env.step(a_sel)
            d = float(terminated or truncated)
            ep_ret += r

            for name in self.order:
                self.rel[name].add(r)

            self.buf_obs.append(obs_t.squeeze(0))
            self.buf_chis.append(chis_t.squeeze(0))
            self.buf_rew.append(float(r))
            self.buf_done.append(d)
            self.buf_val.append(float(val.item()))
            self.buf_act.append(best_idx)

            obs = next_obs
            step += 1
            done = bool(d)

        with torch.no_grad():
            obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            chis_np = np.array([self.rel[n].reliability() for n in self.order], dtype=np.float32)
            chis_t = torch.as_tensor(chis_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_v = self.meta_v(obs_t, chis_t).item()

        self._finish_meta_update(last_value=last_v)
        ret = ep_ret
        self._clear_buf()
        return ret

    def _finish_meta_update(self, last_value=0.0):
        rews = np.array(self.buf_rew, dtype=np.float32)
        dones = np.array(self.buf_done, dtype=np.float32)
        vals = np.array(self.buf_val + [last_value], dtype=np.float32)

        # GAE MC return
        returns = []
        G = last_value
        for r, d in zip(rews[::-1], dones[::-1]):
            G = r + self.gamma * G * (1.0 - d)
            returns.append(G)
        returns = returns[::-1]

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        vals_t    = torch.as_tensor(vals[:-1], dtype=torch.float32, device=self.device)
        adv_t     = returns_t - vals_t
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # === Hard policy loss ===
        obs_batch  = torch.stack(self.buf_obs)
        chis_batch = torch.stack(self.buf_chis)
        act_batch  = torch.as_tensor(self.buf_act, dtype=torch.long, device=self.device)

        logits = self.meta_pi(obs_batch, chis_batch)
        log_probs = torch.log_softmax(logits, dim=-1)
        logp_act  = log_probs[range(len(act_batch)), act_batch]

        policy_loss = -(logp_act * adv_t).mean()
        value_loss = 0.5 * (returns_t - vals_t).pow(2).mean()

        loss = policy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
