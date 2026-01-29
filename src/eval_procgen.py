# src/eval_procgen.py
from __future__ import annotations
import argparse
import yaml
import numpy as np
import torch

from procgen import ProcgenEnv
from gym3 import ToBaselinesVecEnv
from stable_baselines3.common.vec_env import VecMonitor, VecExtractDictObs

from ppo_procgen import Agent


def make_procgen_vec(env_name: str, num_envs: int, level_id: int, distribution_mode: str):
    env = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode,
        start_level=int(level_id),
        num_levels=1,
    )
    venv = ToBaselinesVecEnv(env)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv)
    return venv


def to_torch_obs(obs_np: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) / 255.0
    return x


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(cfg, ckpt_path: str):
    device = cfg["train"]["device"]

    # init env to get n_actions
    tmp = make_procgen_vec(cfg["env"]["name"], cfg["env"]["num_envs"], 0, cfg["env"]["distribution_mode"])
    n_actions = tmp.action_space.n
    tmp.close()

    agent = Agent(n_actions=n_actions).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    test_start = int(cfg["levels"]["test_level_start"])
    test_levels = int(cfg["levels"]["test_levels"])
    episodes_per_level = int(cfg["levels"]["test_episodes_per_level"])

    returns = []

    for i in range(test_levels):
        level_id = test_start + i
        env = make_procgen_vec(cfg["env"]["name"], 1, level_id, cfg["env"]["distribution_mode"])  # eval: 1 env
        for _ in range(episodes_per_level):
            obs = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                obs_t = to_torch_obs(obs, device)
                action, _, _, _ = agent.get_action_and_value(obs_t)
                obs, reward, done_arr, infos = env.step(action.cpu().numpy())
                done = bool(done_arr[0])
                ep_ret += float(reward[0])
            returns.append(ep_ret)
        env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print(f"TEST mean_return={mean_ret:.2f} std={std_ret:.2f} over {len(returns)} episodes")
    return mean_ret


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    evaluate(cfg, args.ckpt)
