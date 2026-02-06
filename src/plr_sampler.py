# src/plr_sampler.py (CORRECTED VERSION)
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from plr_buffer import LevelBuffer, LevelInfo


def _rank_probs(values: List[Tuple[int, float]], alpha: float, temperature: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """
    values: list of (level_id, metric_value). Più alto = più prioritario.
    Rank-based: p(rank) ∝ 1 / rank^alpha, rank=1 è il migliore.
    Temperature: controls sharpness of distribution (lower = more concentrated on high scores)
    """
    # Ordina decrescente (migliore prima)
    values_sorted = sorted(values, key=lambda x: x[1], reverse=True)
    n = len(values_sorted)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    
    # Apply rank-based weighting with temperature
    weights = 1.0 / (ranks ** float(alpha))
    
    # Apply temperature (lower temp = sharper distribution)
    if temperature != 1.0:
        # Use log-space for numerical stability
        log_weights = np.log(weights + 1e-10)
        log_weights = log_weights / temperature
        weights = np.exp(log_weights - log_weights.max())
    
    probs = weights / weights.sum()
    return probs, [lvl for (lvl, _) in values_sorted]


class PLRSampler:
    def __init__(
        self,
        buffer: LevelBuffer,
        train_level_max: int,
        p_new: float,
        alpha: float,
        temperature: float,
        staleness_coef: float,
        warmup_updates: int = 100,
        rng_seed: int = 0
    ):
        """
        Args:
            buffer: LevelBuffer instance
            train_level_max: maximum level ID for new levels
            p_new: probability of sampling a new level (only used during warmup)
            alpha: rank prioritization strength (higher = more uniform)
            temperature: sampling temperature (lower = more focused on high scores)
            staleness_coef: coefficient for staleness bonus (higher = more revisiting of old levels)
            warmup_updates: number of updates to only sample new levels
            rng_seed: random seed
        """
        self.buffer = buffer
        self.train_level_max = int(train_level_max)
        self.p_new = float(p_new)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.staleness_coef = float(staleness_coef)
        self.warmup_updates = int(warmup_updates)
        self.rng = np.random.default_rng(rng_seed)
        self.update_count = 0
        
        # Track sampling statistics
        self.new_level_samples = 0
        self.replay_level_samples = 0

    def sample_level(self, global_step: int) -> Tuple[int, str]:
        """
        Returns (level_id, mode) where mode is "warmup", "new", or "replay".
        """
        self.update_count += 1
        
        # Warmup period: only sample new levels to populate buffer
        if self.update_count <= self.warmup_updates:
            lvl = self._sample_new_level_uniform()
            return lvl, "warmup"
        
        # After warmup: mix of new and replay
        if len(self.buffer) == 0 or self.rng.random() < self.p_new:
            lvl = self._sample_new_level_uniform()
            self.new_level_samples += 1
            return lvl, "new"
        
        lvl = self._sample_replay_level(global_step)
        self.replay_level_samples += 1
        return lvl, "replay"

    def _sample_new_level_uniform(self) -> int:
        """Sample a new level uniformly at random from [0, train_level_max)"""
        return int(self.rng.integers(0, self.train_level_max))

    def _sample_replay_level(self, global_step: int) -> int:
        """
        Sample a level from the buffer using score + staleness.
        
        Following the original PLR implementation:
        1. Calculate combined score = base_score + staleness_bonus
        2. Apply rank-based prioritization with temperature
        3. Sample from the resulting distribution
        """
        infos: List[LevelInfo] = self.buffer.get_all()
        
        if len(infos) == 0:
            # Fallback: sample new level
            return self._sample_new_level_uniform()
        
        # Calculate combined score with staleness bonus
        combined_scores = []
        for li in infos:
            staleness = float(global_step - li.last_seen_step)
            # Staleness bonus: older levels get higher bonus
            staleness_bonus = self.staleness_coef * staleness
            combined_score = li.score + staleness_bonus
            combined_scores.append((li.level_id, combined_score))
        
        # Apply rank-based prioritization with temperature
        probs, levels = _rank_probs(combined_scores, self.alpha, self.temperature)
        
        # Sample
        chosen_idx = int(self.rng.choice(len(levels), p=probs))
        return int(levels[chosen_idx])
    
    def get_sampling_stats(self) -> dict:
        """Return statistics about sampling behavior"""
        total = self.new_level_samples + self.replay_level_samples
        return {
            "new_samples": self.new_level_samples,
            "replay_samples": self.replay_level_samples,
            "replay_ratio": self.replay_level_samples / max(1, total),
            "buffer_size": len(self.buffer),
            "update_count": self.update_count,
            "warmup_complete": self.update_count > self.warmup_updates
        }