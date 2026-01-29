# src/train_plr_procgen.py
from __future__ import annotations
import os
import time
import yaml
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.optim as optim
import gymnasium as gym

from procgen import ProcgenEnv
from gym3 import ToBaselinesVecEnv
from stable_baselines3.common.vec_env import VecMonitor, VecExtractDictObs

from plr_buffer import LevelBuffer
from plr_sampler import PLRSampler
from ppo_procgen import Agent, PPOHParams, compute_gae, ppo_update


def make_procgen_vec(env_name: str, num_envs: int, level_id: int, distribution_mode: str):
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode,
        start_level=int(level_id),
        num_levels=1,
    )

    # Procgen può restituire obs dict con chiave "rgb" oppure direttamente array.
    # Se è Dict, estraiamo "rgb".
    if isinstance(venv.observation_space, gym.spaces.Dict):
        venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(venv)
    return venv


def to_torch_obs(obs_np: np.ndarray, device: str) -> torch.Tensor:
    # obs: (N, 64, 64, 3) uint8 -> (N, 3, 64, 64) float32 in [0,1]
    x = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) / 255.0
    return x


def unwrap_obs(obs):
    # Procgen può restituire dict con chiave "rgb"
    if isinstance(obs, dict):
        return obs["rgb"]
    return obs


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)

    # seeds
    seed = int(cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = cfg["train"]["device"]
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    run_name = f'{cfg["env"]["name"]}_plr_{int(time.time())}'
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # PLR components
    buffer = LevelBuffer(
        max_size=int(cfg["plr"]["buffer_size"]),
        score_ema_beta=float(cfg["plr"]["score_ema_beta"]),
        rng=random.Random(seed),
    )
    sampler = PLRSampler(
        buffer=buffer,
        train_level_max=int(cfg["levels"]["train_level_max"]),
        p_new=float(cfg["plr"]["p_new"]),
        alpha=float(cfg["plr"]["alpha"]),
        rho=float(cfg["plr"]["rho"]),
        rng_seed=seed,
    )

    # bootstrap env to get action space
    tmp_env = make_procgen_vec(cfg["env"]["name"], cfg["env"]["num_envs"], level_id=0, distribution_mode=cfg["env"]["distribution_mode"])
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    agent = Agent(n_actions=n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(cfg["ppo"]["learning_rate"]), eps=1e-5)
    h = PPOHParams(
        learning_rate=float(cfg["ppo"]["learning_rate"]),
        gamma=float(cfg["ppo"]["gamma"]),
        gae_lambda=float(cfg["ppo"]["gae_lambda"]),
        clip_coef=float(cfg["ppo"]["clip_coef"]),
        ent_coef=float(cfg["ppo"]["ent_coef"]),
        vf_coef=float(cfg["ppo"]["vf_coef"]),
        max_grad_norm=float(cfg["ppo"]["max_grad_norm"]),
        update_epochs=int(cfg["ppo"]["update_epochs"]),
        minibatches=int(cfg["ppo"]["minibatches"]),
    )

    total_updates = int(cfg["train"]["total_updates"])
    num_steps = int(cfg["train"]["num_steps"])
    num_envs = int(cfg["env"]["num_envs"])
    save_every = int(cfg["train"]["save_every"])

    global_step = 0
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("update,global_step,mode,level_id,score,mean_ep_return\n")

    for update in range(1, total_updates + 1):
        level_id, mode = sampler.sample_level(global_step)

        env = make_procgen_vec(
            env_name=cfg["env"]["name"],
            num_envs=num_envs,
            level_id=level_id,
            distribution_mode=cfg["env"]["distribution_mode"],
        )
        obs = env.reset()
        obs = unwrap_obs(obs)

        # rollout storage
        obs_buf = np.zeros((num_steps, num_envs, 64, 64, 3), dtype=np.uint8)
        actions_buf = np.zeros((num_steps, num_envs), dtype=np.int64)
        logprobs_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        rewards_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        dones_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        values_buf = np.zeros((num_steps, num_envs), dtype=np.float32)

        ep_returns = []

        next_done = np.zeros(num_envs, dtype=np.float32)

        for t in range(num_steps):
            global_step += num_envs

            obs_buf[t] = obs
            dones_buf[t] = next_done

            obs_t = to_torch_obs(obs, device)
            with torch.no_grad():
                action_t, logp_t, _, value_t = agent.get_action_and_value(obs_t)

            actions = action_t.cpu().numpy()
            next_obs, reward, done, infos = env.step(actions)
            next_obs = unwrap_obs(next_obs)

            actions_buf[t] = actions
            logprobs_buf[t] = logp_t.cpu().numpy()
            rewards_buf[t] = reward
            values_buf[t] = value_t.cpu().numpy()

            # VecMonitor mette info["episode"] quando un episodio finisce
            for info in infos:
                if "episode" in info:
                    ep_returns.append(info["episode"]["r"])

            obs = next_obs
            next_done = done.astype(np.float32)

        # bootstrap value
        with torch.no_grad():
            next_value = agent.get_action_and_value(to_torch_obs(obs, device))[3]  # value
        next_value_t = next_value  # (N,)

        # to torch
        rewards_t = torch.from_numpy(rewards_buf).to(device)
        dones_t = torch.from_numpy(dones_buf).to(device)
        values_t = torch.from_numpy(values_buf).to(device)

        adv_t, ret_t = compute_gae(
            rewards=rewards_t,
            dones=dones_t,
            values=values_t,
            next_value=next_value_t,
            gamma=h.gamma,
            gae_lambda=h.gae_lambda,
        )

        # PLR score: mean(abs(adv)) (paper-style proxy)
        score = float(adv_t.abs().mean().item())
        buffer.update(level_id=level_id, score=score, global_step=global_step)

        # flatten batch
        b_obs = torch.from_numpy(obs_buf).to(device)          # (T,N,H,W,C) uint8
        b_obs = b_obs.permute(0, 1, 4, 2, 3).float() / 255.0  # (T,N,3,64,64)
        b_obs = b_obs.reshape(-1, 3, 64, 64)

        b_actions = torch.from_numpy(actions_buf).to(device).reshape(-1)
        b_logprobs = torch.from_numpy(logprobs_buf).to(device).reshape(-1)
        b_adv = adv_t.reshape(-1)
        b_ret = ret_t.reshape(-1)
        b_val = values_t.reshape(-1)

        ppo_update(
            agent=agent,
            optimizer=optimizer,
            h=h,
            obs=b_obs,
            actions=b_actions,
            logprobs=b_logprobs,
            advantages=b_adv,
            returns=b_ret,
            values=b_val,
        )

        env.close()

        mean_ep_return = float(np.mean(ep_returns)) if len(ep_returns) else float("nan")
        with open(log_path, "a") as f:
            f.write(f"{update},{global_step},{mode},{level_id},{score},{mean_ep_return}\n")

        if update % 50 == 0:
            print(f"[{update}/{total_updates}] mode={mode} level={level_id} score={score:.4f} mean_ep_return={mean_ep_return}")

        if update % save_every == 0:
            ckpt_path = os.path.join(run_dir, f"ckpt_{update}.pt")
            torch.save({"agent": agent.state_dict(), "update": update, "config": cfg}, ckpt_path)

    final_path = os.path.join(run_dir, "final.pt")
    torch.save({"agent": agent.state_dict(), "update": total_updates, "config": cfg}, final_path)
    print("Saved:", final_path)
    print("Logs:", log_path)


if __name__ == "__main__":
    main()
