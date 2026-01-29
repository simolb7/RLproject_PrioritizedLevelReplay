# src/plr_buffer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class LevelInfo:
    level_id: int
    score: float
    last_seen_step: int
    visits: int = 0


class LevelBuffer:
    """
    Memorizza livelli visti e metadati (score/staleness).
    Eviction semplice: rimuove livelli con score più basso (più "inutile") quando pieno.
    """
    def __init__(self, max_size: int, score_ema_beta: float = 0.9, rng: Optional[random.Random] = None):
        self.max_size = max_size
        self.beta = float(score_ema_beta)
        self.rng = rng or random.Random()
        self._data: Dict[int, LevelInfo] = {}

    def __len__(self) -> int:
        return len(self._data)

    def has(self, level_id: int) -> bool:
        return level_id in self._data

    def get_all(self) -> List[LevelInfo]:
        return list(self._data.values())

    def update(self, level_id: int, score: float, global_step: int) -> None:
        score = float(score)
        if level_id in self._data:
            li = self._data[level_id]
            li.score = self.beta * li.score + (1.0 - self.beta) * score
            li.last_seen_step = global_step
            li.visits += 1
        else:
            self._data[level_id] = LevelInfo(
                level_id=level_id,
                score=score,
                last_seen_step=global_step,
                visits=1
            )
            self._evict_if_needed()

    def staleness(self, level_id: int, global_step: int) -> int:
        li = self._data[level_id]
        return max(0, global_step - li.last_seen_step)

    def _evict_if_needed(self) -> None:
        if len(self._data) <= self.max_size:
            return

        # Eviction: rimuovi un po' dei peggiori per score (semplice, efficace per progetto d'esame).
        # Se vuoi, puoi cambiarlo con FIFO o "least recently used".
        n_remove = len(self._data) - self.max_size
        items = sorted(self._data.values(), key=lambda x: x.score)  # score crescente
        for i in range(n_remove):
            del self._data[items[i].level_id]
