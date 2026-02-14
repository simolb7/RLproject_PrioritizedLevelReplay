from __future__ import annotations
from typing import List, Tuple
import numpy as np

from plr_buffer import LevelBuffer, LevelInfo


def _rank_probs(values: List[Tuple[int, float]], alpha: float, temperature: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    values_sorted = sorted(values, key=lambda x: x[1], reverse=True)
    n = len(values_sorted)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    
    weights = 1.0 / (ranks ** float(alpha))
    
    if temperature != 1.0:
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
        temperature_schedule: str = "constant", 
        score_transform: str = "normalize", 
        rng_seed: int = 0
    ):
        self.buffer = buffer
        self.train_level_max = int(train_level_max)
        self.p_new = float(p_new)
        self.alpha = float(alpha)
        self.base_temperature = float(temperature)
        self.temperature = float(temperature)
        self.staleness_coef = float(staleness_coef)
        self.warmup_updates = int(warmup_updates)
        self.temperature_schedule = temperature_schedule
        self.score_transform = score_transform
        self.rng = np.random.default_rng(rng_seed)
        self.update_count = 0
        

        self.new_level_samples = 0
        self.replay_level_samples = 0
        self.score_stats = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}

    def sample_level(self, global_step: int) -> Tuple[int, str]:

        self.update_count += 1
        self._update_temperature()
        
        if self.update_count <= self.warmup_updates:
            lvl = self._sample_new_level_uniform()
            return lvl, "warmup"
        
        if len(self.buffer) == 0 or self.rng.random() < self.p_new:
            lvl = self._sample_new_level_uniform()
            self.new_level_samples += 1
            return lvl, "new"
        
        lvl = self._sample_replay_level(global_step)
        self.replay_level_samples += 1
        return lvl, "replay"

    def _update_temperature(self):

        if self.temperature_schedule == "constant":
            return
        
        if self.temperature_schedule == "linear_decay":
            progress = min(1.0, self.update_count / (self.warmup_updates * 10))
            self.temperature = self.base_temperature * (1.0 - 0.5 * progress)
        
        elif self.temperature_schedule == "adaptive":
            if len(self.buffer) > 10:
                infos = self.buffer.get_all()
                scores = [li.score for li in infos]
                score_std = float(np.std(scores))
                diversity_factor = max(0.5, min(2.0, 1.0 / (score_std + 0.1)))
                self.temperature = self.base_temperature * diversity_factor

    def _sample_new_level_uniform(self) -> int:
        return int(self.rng.integers(0, self.train_level_max))

    def _update_score_stats(self, scores: List[float]):
        if len(scores) == 0:
            return
        
        scores_arr = np.array(scores)
        self.score_stats = {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)) + 1e-8,
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)) + 1e-8
        }

    def _transform_score(self, score: float) -> float:
        if self.score_transform == "none":
            return score
        
        elif self.score_transform == "normalize":
            score_range = self.score_stats["max"] - self.score_stats["min"]
            return (score - self.score_stats["min"]) / max(score_range, 1e-8)
        
        elif self.score_transform == "standardize":
            return (score - self.score_stats["mean"]) / self.score_stats["std"]
        
        return score

    def _sample_replay_level(self, global_step: int) -> int:
        infos: List[LevelInfo] = self.buffer.get_all()
        
        if len(infos) == 0:
            return self._sample_new_level_uniform()
        
        base_scores = [li.score for li in infos]
        self._update_score_stats(base_scores)
        
        max_staleness = max(1.0, float(global_step - min(li.last_seen_step for li in infos)))
        
        combined_scores = []
        for li in infos:
            transformed_score = self._transform_score(li.score)
            
            staleness = float(global_step - li.last_seen_step)
            normalized_staleness = staleness / max_staleness
            staleness_bonus = self.staleness_coef * normalized_staleness
            
            combined_score = transformed_score + staleness_bonus
            combined_scores.append((li.level_id, combined_score))
        
        probs, levels = _rank_probs(combined_scores, self.alpha, self.temperature)
        
        chosen_idx = int(self.rng.choice(len(levels), p=probs))
        return int(levels[chosen_idx])
    
    def get_sampling_stats(self) -> dict:
        total = self.new_level_samples + self.replay_level_samples
        return {
            "new_samples": self.new_level_samples,
            "replay_samples": self.replay_level_samples,
            "replay_ratio": self.replay_level_samples / max(1, total),
            "buffer_size": len(self.buffer),
            "update_count": self.update_count,
            "warmup_complete": self.update_count > self.warmup_updates,
            "current_temperature": self.temperature,
            "score_mean": self.score_stats["mean"],
            "score_std": self.score_stats["std"]
        }