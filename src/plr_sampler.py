# src/plr_sampler.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from plr_buffer import LevelBuffer, LevelInfo


def _rank_probs(values: List[Tuple[int, float]], alpha: float) -> np.ndarray:
    """
    values: list of (level_id, metric_value). Più alto = più prioritario.
    Rank-based: p(rank) ∝ 1 / rank^alpha, rank=1 è il migliore.
    """
    # Ordina decrescente (migliore prima)
    values_sorted = sorted(values, key=lambda x: x[1], reverse=True)
    n = len(values_sorted)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** float(alpha))
    probs = weights / weights.sum()
    return probs, [lvl for (lvl, _) in values_sorted]


class PLRSampler:
    def __init__(
        self,
        buffer: LevelBuffer,
        train_level_max: int,
        p_new: float,
        alpha: float,
        rho: float,
        rng_seed: int = 0
    ):
        self.buffer = buffer
        self.train_level_max = int(train_level_max)
        self.p_new = float(p_new)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.rng = np.random.default_rng(rng_seed)
        self._next_new_level = 0  # semplice "curriculum" su livelli nuovi: 0,1,2,...

    def sample_level(self, global_step: int) -> Tuple[int, str]:
        """
        Ritorna (level_id, mode) dove mode è "new" o "replay".
        """
        if len(self.buffer) == 0 or self.rng.random() < self.p_new:
            lvl = self._sample_new_level()
            return lvl, "new"
        lvl = self._sample_replay_level(global_step)
        return lvl, "replay"

    def _sample_new_level(self) -> int:
        # Nuovi livelli: percorriamo [0, train_level_max) in modo deterministico (poi random).
        if self._next_new_level < self.train_level_max:
            lvl = self._next_new_level
            self._next_new_level += 1
            return lvl
        return int(self.rng.integers(0, self.train_level_max))

    def _sample_replay_level(self, global_step: int) -> int:
        infos: List[LevelInfo] = self.buffer.get_all()

        # Score priorities
        score_values = [(li.level_id, li.score) for li in infos]
        p_score, levels_by_score = _rank_probs(score_values, self.alpha)

        # Staleness priorities (più stantio = più prioritario)
        stale_values = [(li.level_id, float(global_step - li.last_seen_step)) for li in infos]
        p_stale, levels_by_stale = _rank_probs(stale_values, self.alpha)

        # Mettiamo le due distribuzioni nello stesso ordine di livelli, usando un mapping.
        idx_score = {lvl: i for i, lvl in enumerate(levels_by_score)}
        idx_stale = {lvl: i for i, lvl in enumerate(levels_by_stale)}

        levels = levels_by_score  # base order
        p_stale_aligned = np.array([p_stale[idx_stale[lvl]] for lvl in levels], dtype=np.float64)
        p_score_aligned = p_score.astype(np.float64)

        p_mix = (1.0 - self.rho) * p_score_aligned + self.rho * p_stale_aligned
        p_mix = p_mix / p_mix.sum()

        chosen = int(self.rng.choice(len(levels), p=p_mix))
        return int(levels[chosen])
