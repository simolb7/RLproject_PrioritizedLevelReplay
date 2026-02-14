# src/plr_buffer.py (CORRECTED VERSION)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import numpy as np


@dataclass
class LevelInfo:
    level_id: int
    score: float
    last_seen_step: int
    visits: int = 0
    first_seen_step: int = 0


class LevelBuffer:
    def __init__(
        self, 
        max_size: int, 
        score_ema_beta: float = 0.99, 
        eviction_mode: str = "score_staleness",
        rng: Optional[random.Random] = None
    ):
        self.max_size = max_size
        self.beta = float(score_ema_beta)
        self.eviction_mode = eviction_mode
        self.rng = rng or random.Random()
        self._data: Dict[int, LevelInfo] = {}

    def __len__(self) -> int:
        return len(self._data)

    def has(self, level_id: int) -> bool:
        return level_id in self._data

    def get_all(self) -> List[LevelInfo]:
        return list(self._data.values())
    
    def get(self, level_id: int) -> Optional[LevelInfo]:
        return self._data.get(level_id)

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
                first_seen_step=global_step,
                visits=1
            )
            self._evict_if_needed(global_step)

    def staleness(self, level_id: int, global_step: int) -> int:
        if level_id not in self._data:
            return 0
        li = self._data[level_id]
        return max(0, global_step - li.last_seen_step)

    def _evict_if_needed(self, global_step: int) -> None:

        if len(self._data) <= self.max_size:
            return

        n_remove = len(self._data) - self.max_size
        items = list(self._data.values())
        
        if self.eviction_mode == "score":
            
            items_sorted = sorted(items, key=lambda x: x.score)
            
        elif self.eviction_mode == "lru":
            
            items_sorted = sorted(items, key=lambda x: global_step - x.last_seen_step, reverse=True)
            
        elif self.eviction_mode == "score_staleness":
        
            def eviction_score(li: LevelInfo) -> float:
                staleness = global_step - li.last_seen_step
                max_staleness = max(1, global_step - min(x.first_seen_step for x in items))
                max_score = max(x.score for x in items) + 1e-6
                
                norm_staleness = staleness / max_staleness
                norm_score = li.score / max_score
                
                return norm_score * 0.7 - norm_staleness * 0.3
            
            items_sorted = sorted(items, key=eviction_score)
            
        elif self.eviction_mode == "random":
            self.rng.shuffle(items)
            items_sorted = items
            
        else:
            items_sorted = sorted(items, key=lambda x: x.score)
        
        for i in range(n_remove):
            del self._data[items_sorted[i].level_id]
    
    def get_statistics(self, global_step: int) -> dict:
        if len(self._data) == 0:
            return {
                "size": 0,
                "mean_score": 0.0,
                "std_score": 0.0,
                "mean_staleness": 0.0,
                "mean_visits": 0.0
            }
        
        infos = list(self._data.values())
        scores = [li.score for li in infos]
        staleness = [global_step - li.last_seen_step for li in infos]
        visits = [li.visits for li in infos]
        
        return {
            "size": len(self._data),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "mean_staleness": float(np.mean(staleness)),
            "max_staleness": float(np.max(staleness)),
            "mean_visits": float(np.mean(visits)),
            "total_visits": sum(visits)
        }