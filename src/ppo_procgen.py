# src/ppo_procgen.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Agent(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        # Input Procgen: (B, 3, 64, 64) uint8 -> float in [0,1]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # calcola dimensione
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            n_flat = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU())
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(self.cnn(x))
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


@dataclass
class PPOHParams:
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    update_epochs: int
    minibatches: int


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    """
    rewards: (T, N)
    dones:   (T, N)  1.0 se done, 0.0 altrimenti
    values:  (T, N)
    next_value: (N,)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones[t]
        nextvalues = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_update(
    agent: Agent,
    optimizer: optim.Optimizer,
    h: PPOHParams,
    obs: torch.Tensor,          # (T*N, 3, 64, 64)
    actions: torch.Tensor,      # (T*N,)
    logprobs: torch.Tensor,     # (T*N,)
    advantages: torch.Tensor,   # (T*N,)
    returns: torch.Tensor,      # (T*N,)
    values: torch.Tensor,       # (T*N,)
):
    Tn = obs.shape[0]
    batch_size = Tn
    minibatch_size = batch_size // h.minibatches

    # normalizza advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(h.update_epochs):
        idx = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            mb = idx[start:start + minibatch_size]

            logits, new_values = agent.forward(obs[mb])
            dist = torch.distributions.Categorical(logits=logits)
            new_logprob = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            logratio = new_logprob - logprobs[mb]
            ratio = logratio.exp()

            # policy loss (clipped)
            mb_adv = advantages[mb]
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - h.clip_coef, 1.0 + h.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # value loss (clipped)
            v_loss_unclipped = (new_values - returns[mb]) ** 2
            v_clipped = values[mb] + torch.clamp(new_values - values[mb], -h.clip_coef, h.clip_coef)
            v_loss_clipped = (v_clipped - returns[mb]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            loss = pg_loss - h.ent_coef * entropy + h.vf_coef * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), h.max_grad_norm)
            optimizer.step()
