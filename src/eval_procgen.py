# src/eval_procgen.py
from __future__ import annotations
import argparse
import yaml
import numpy as np
import torch
import time
from pathlib import Path
from PIL import Image

from procgen import ProcgenEnv
from gym3 import ToBaselinesVecEnv
from stable_baselines3.common.vec_env import VecMonitor, VecExtractDictObs

from ppo_procgen import Agent
import pygame


def make_procgen_vec(env_name: str, num_envs: int, level_id: int, distribution_mode: str):
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode,
        start_level=int(level_id),
        num_levels=1,
    )
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


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(cfg, ckpt_path: str, env_name: str, render: bool = False, single_level: int = None, save_gif: str = None):
    device = cfg["train"]["device"]

    # init env to get n_actions
    tmp = make_procgen_vec(env_name, cfg["env"]["num_envs"], 0, cfg["env"]["distribution_mode"])
    n_actions = tmp.action_space.n
    tmp.close()

    agent = Agent(n_actions=n_actions).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    # Setup pygame for rendering if requested
    screen = None
    clock = None
    if render or save_gif:
        pygame.init()
        if render:
            screen = pygame.display.set_mode((512, 512))
            pygame.display.set_caption(f"Procgen - {env_name}")
        clock = pygame.time.Clock()

    # Determine which levels to test
    if single_level is not None:
        test_levels_list = [single_level]
        episodes_per_level = int(cfg["levels"]["test_episodes_per_level"])
    else:
        test_start = int(cfg["levels"]["test_level_start"])
        test_levels = int(cfg["levels"]["test_levels"])
        test_levels_list = [test_start + i for i in range(test_levels)]
        episodes_per_level = int(cfg["levels"]["test_episodes_per_level"])

    returns = []

    for level_id in test_levels_list:
        env = make_procgen_vec(env_name, 1, level_id, cfg["env"]["distribution_mode"])
        
        for ep_idx in range(episodes_per_level):
            obs = env.reset()
            obs = unwrap_obs(obs)
            done = False
            ep_ret = 0.0
            step_count = 0
            
            # Storage for GIF frames
            frames = [] if save_gif else None
            
            while not done:
                obs_t = to_torch_obs(obs, device)
                action, _, _, _ = agent.get_action_and_value(obs_t)
                obs, reward, done_arr, infos = env.step(action.cpu().numpy())
                obs = unwrap_obs(obs)
                done = bool(done_arr[0])
                ep_ret += float(reward[0])
                step_count += 1
                
                # Rendering and/or saving frames
                if render or save_gif:
                    # Get frame from observation (Procgen returns RGB in obs)
                    frame = obs[0] if len(obs.shape) == 4 else obs
                    
                    # Create surface for rendering
                    surf = pygame.Surface((64, 64))
                    pygame.surfarray.blit_array(surf, np.transpose(frame, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, (512, 512))
                    
                    # Add info overlay
                    font = pygame.font.Font(None, 36)
                    text = font.render(f"Lvl:{level_id} Ep:{ep_idx+1} Ret:{ep_ret:.1f} Step:{step_count}", True, (255, 255, 255))
                    text_shadow = font.render(f"Lvl:{level_id} Ep:{ep_idx+1} Ret:{ep_ret:.1f} Step:{step_count}", True, (0, 0, 0))
                    surf.blit(text_shadow, (12, 12))
                    surf.blit(text, (10, 10))
                    
                    # Display on screen if rendering
                    if render and screen is not None:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                env.close()
                                pygame.quit()
                                return np.mean(returns) if returns else 0.0
                        
                        screen.blit(surf, (0, 0))
                        pygame.display.flip()
                        clock.tick(30)  # 30 FPS
                    
                    # Save frame for GIF
                    if save_gif:
                        # Convert pygame surface to PIL Image
                        frame_array = pygame.surfarray.array3d(surf)
                        frame_array = np.transpose(frame_array, (1, 0, 2))
                        frames.append(Image.fromarray(frame_array.astype(np.uint8)))
            
            returns.append(ep_ret)
            if single_level is not None:
                print(f"Level {level_id}, Episode {ep_idx+1}/{episodes_per_level}: return={ep_ret:.2f}, steps={step_count}")
            
            # Save GIF if requested
            if save_gif and frames:
                output_path = Path(save_gif)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Generate filename
                if output_path.is_dir():
                    gif_filename = output_path / f"{env_name}_level{level_id}_ep{ep_idx+1}.gif"
                else:
                    gif_filename = output_path
                
                print(f"Saving GIF to {gif_filename}...")
                frames[0].save(
                    gif_filename,
                    save_all=True,
                    append_images=frames[1:],
                    duration=33,  # ~30 FPS
                    loop=0
                )
        
        env.close()

    if render or save_gif:
        pygame.quit()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print(f"\nTEST mean_return={mean_ret:.2f} std={std_ret:.2f} over {len(returns)} episodes")
    return mean_ret


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="coinrun", choices=["coinrun", "bigfish", "chaser", "starpilot", "jumper"], help="Environment name")
    ap.add_argument("--config", default="configs/coinrun.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--render", action="store_true", help="Enable visual rendering with pygame")
    ap.add_argument("--level", type=int, default=None, help="Evaluate on a single specific level (overrides config test levels)")
    ap.add_argument("--save-gif", type=str, default=None, help="Save rendering as GIF. Provide path (file or directory)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    evaluate(cfg, args.ckpt, args.env, render=args.render, single_level=args.level, save_gif=args.save_gif)