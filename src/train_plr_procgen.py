from __future__ import annotations
import os
import time
import yaml
import random
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
import gymnasium as gym

from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import VecMonitor, VecExtractDictObs

from plr_buffer import LevelBuffer
from plr_sampler import PLRSamplerImproved
from ppo_procgen import Agent, PPOHParams, compute_gae, ppo_update


def make_procgen_vec_multi(env_name: str, num_envs: int, level_ids: list, distribution_mode: str):
    """
    Create vectorized env with DIFFERENT levels per env.
    This is crucial for PLR to get diverse score estimates.
    """
    envs = []
    for level_id in level_ids:
        venv = ProcgenEnv(
            num_envs=1,
            env_name=env_name,
            distribution_mode=distribution_mode,
            start_level=int(level_id),
            num_levels=1,
        )
        
        if isinstance(venv.observation_space, gym.spaces.Dict):
            venv = VecExtractDictObs(venv, "rgb")
        
        envs.append(venv)
    
    # Stack environments
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env(procgen_env):
        def _init():
            return procgen_env
        return _init
    
    # Note: This is a simplification - in practice you'd use a proper wrapper
    # For now, we'll use the single-level approach but with better score calculation
    return None


def make_procgen_vec(env_name: str, num_envs: int, level_id: int, distribution_mode: str):
    """Standard single-level vectorized environment."""
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode,
        start_level=int(level_id),
        num_levels=1,
    )

    if isinstance(venv.observation_space, gym.spaces.Dict):
        venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(venv)
    return venv


def to_torch_obs(obs_np: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) / 255.0
    return x


def unwrap_obs(obs):
    if isinstance(obs, dict):
        return obs["rgb"]
    return obs


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def calculate_level_score(
    returns: torch.Tensor,
    values: torch.Tensor,
    method: str = "value_l1"
) -> float:
    """
    Calculate score for a level using different methods.
    
    Args:
        returns: (T, N) bootstrapped returns
        values: (T, N) value predictions
        method: scoring method
    
    Returns:
        Scalar score (higher = more learning potential)
    """
    if method == "value_l1":
        # L1 value loss - paper's preferred metric
        return float((returns - values).abs().mean().item())
    
    elif method == "value_l2":
        # L2 value loss
        return float(((returns - values) ** 2).mean().sqrt().item())
    
    elif method == "gae":
        # Use GAE magnitude as proxy (advantages = returns - values for unbiased estimator)
        advantages = returns - values
        return float(advantages.abs().mean().item())
    
    elif method == "max_gae":
        # Max absolute GAE (for extremely hard levels)
        advantages = returns - values
        return float(advantages.abs().max().item())
    
    else:
        # Default: value_l1
        return float((returns - values).abs().mean().item())


def main(config_path: str = "configs/default.yaml", env_name: str = "coinrun"):
    cfg = load_config(config_path)

    # Seeds
    seed = int(cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    run_name = f'{env_name}_plr_improved_{int(time.time())}'
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # PLR components with improved sampler
    buffer = LevelBuffer(
        max_size=int(cfg["plr"]["buffer_size"]),
        score_ema_beta=float(cfg["plr"].get("score_ema_beta", 0.99)),
        eviction_mode=cfg["plr"].get("eviction_mode", "score_staleness"),
        rng=random.Random(seed),
    )
    
    sampler = PLRSamplerImproved(
        buffer=buffer,
        train_level_max=int(cfg["levels"]["train_level_max"]),
        p_new=float(cfg["plr"]["p_new"]),
        alpha=float(cfg["plr"].get("alpha", 1.0)),
        temperature=float(cfg["plr"].get("temperature", 0.1)),
        staleness_coef=float(cfg["plr"].get("staleness_coef", 0.3)),  # Increased default
        warmup_updates=int(cfg["plr"].get("warmup_updates", 100)),
        temperature_schedule=cfg["plr"].get("temperature_schedule", "constant"),
        score_transform=cfg["plr"].get("score_transform", "normalize"),
        rng_seed=seed,
    )

    # Bootstrap env
    tmp_env = make_procgen_vec(
        env_name, 
        cfg["env"]["num_envs"], 
        level_id=0, 
        distribution_mode=cfg["env"]["distribution_mode"]
    )
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    agent = Agent(n_actions=n_actions).to(device)
    optimizer = optim.Adam(
        agent.parameters(), 
        lr=float(cfg["ppo"]["learning_rate"]), 
        eps=1e-5
    )
    
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
    score_method = cfg["plr"].get("score_metric", "value_l1")

    global_step = 0
    
    # Enhanced logging
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("update,global_step,mode,level_id,score,mean_ep_return,buffer_size,replay_ratio,temperature\n")
    
    stats_log_path = os.path.join(run_dir, "buffer_stats.csv")
    with open(stats_log_path, "w") as f:
        f.write("update,buffer_size,mean_score,std_score,mean_staleness,replay_ratio\n")

    # Level visit tracking for diagnostics
    level_visits = defaultdict(int)

    for update in range(1, total_updates + 1):
        level_id, mode = sampler.sample_level(global_step)
        level_visits[level_id] += 1

        env = make_procgen_vec(
            env_name=env_name,
            num_envs=num_envs,
            level_id=level_id,
            distribution_mode=cfg["env"]["distribution_mode"],
        )
        obs = env.reset()
        obs = unwrap_obs(obs)

        # Rollout storage
        obs_buf = np.zeros((num_steps, num_envs, 64, 64, 3), dtype=np.uint8)
        actions_buf = np.zeros((num_steps, num_envs), dtype=np.int64)
        logprobs_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        rewards_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        dones_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        values_buf = np.zeros((num_steps, num_envs), dtype=np.float32)

        ep_returns = []
        next_done = np.zeros(num_envs, dtype=np.float32)

        # Collect rollout
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

            for info in infos:
                if "episode" in info:
                    ep_returns.append(info["episode"]["r"])

            obs = next_obs
            next_done = done.astype(np.float32)

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_action_and_value(to_torch_obs(obs, device))[3]

        rewards_t = torch.from_numpy(rewards_buf).to(device)
        dones_t = torch.from_numpy(dones_buf).to(device)
        values_t = torch.from_numpy(values_buf).to(device)

        adv_t, ret_t = compute_gae(
            rewards=rewards_t,
            dones=dones_t,
            values=values_t,
            next_value=next_value,
            gamma=h.gamma,
            gae_lambda=h.gae_lambda,
        )

        # Calculate score using selected method
        score = calculate_level_score(ret_t, values_t, method=score_method)
        buffer.update(level_id=level_id, score=score, global_step=global_step)

        # Flatten batch
        b_obs = torch.from_numpy(obs_buf).to(device)
        b_obs = b_obs.permute(0, 1, 4, 2, 3).float() / 255.0
        b_obs = b_obs.reshape(-1, 3, 64, 64)

        b_actions = torch.from_numpy(actions_buf).to(device).reshape(-1)
        b_logprobs = torch.from_numpy(logprobs_buf).to(device).reshape(-1)
        b_adv = adv_t.reshape(-1)
        b_ret = ret_t.reshape(-1)
        b_val = values_t.reshape(-1)

        # PPO update
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

        # Logging
        mean_ep_return = float(np.mean(ep_returns)) if len(ep_returns) else float("nan")
        sampling_stats = sampler.get_sampling_stats()
        
        with open(log_path, "a") as f:
            f.write(f"{update},{global_step},{mode},{level_id},{score:.6f},{mean_ep_return},"
                   f"{sampling_stats['buffer_size']},{sampling_stats['replay_ratio']:.3f},"
                   f"{sampling_stats['current_temperature']:.3f}\n")

        # Buffer statistics every 10 updates
        if update % 10 == 0:
            buffer_stats = buffer.get_statistics(global_step)
            with open(stats_log_path, "a") as f:
                f.write(f"{update},{buffer_stats['size']},{buffer_stats['mean_score']:.6f},"
                       f"{buffer_stats['std_score']:.6f},{buffer_stats['mean_staleness']:.1f},"
                       f"{sampling_stats['replay_ratio']:.3f}\n")

        # Console output
        if update % 50 == 0:
            print(f"[{update}/{total_updates}] mode={mode} lvl={level_id} score={score:.4f} "
                  f"ep_ret={mean_ep_return:.2f} buf={len(buffer)} "
                  f"replay={sampling_stats['replay_ratio']:.2f} temp={sampling_stats['current_temperature']:.2f}")

        # Checkpoint
        if update % save_every == 0:
            ckpt_path = os.path.join(run_dir, f"ckpt_{update}.pt")
            torch.save({
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "config": cfg,
                "buffer_stats": buffer.get_statistics(global_step),
                "sampling_stats": sampling_stats
            }, ckpt_path)

    # Final save
    final_path = os.path.join(run_dir, "final.pt")
    torch.save({
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": total_updates,
        "config": cfg,
        "buffer_stats": buffer.get_statistics(global_step),
        "level_visits": dict(level_visits)
    }, final_path)
    
    print(f"\nTraining complete!")
    print(f"Model saved: {final_path}")
    print(f"Logs: {log_path}")
    print(f"Buffer stats: {stats_log_path}")
    print(f"\nFinal buffer size: {len(buffer)}")
    print(f"Unique levels visited: {len(level_visits)}")
    print(f"Total replay ratio: {sampling_stats['replay_ratio']:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="coinrun", choices=["coinrun", "bigfish", "chaser"], help="Environment name")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(args.config, args.env)