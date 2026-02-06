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
    first_seen_step: int = 0  # Tracking when level was first added


class LevelBuffer:
    """
    Stores seen levels and metadata (score/staleness).
    Improved eviction: combination of score, staleness, and randomness
    to avoid removing potentially useful levels.
    """
    def __init__(
        self, 
        max_size: int, 
        score_ema_beta: float = 0.99,  # Changed default to match original
        eviction_mode: str = "score_staleness",  # "score", "lru", "score_staleness", "random"
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
        """Update or add a level to the buffer."""
        score = float(score)
        
        if level_id in self._data:
            # Update existing level
            li = self._data[level_id]
            # EMA update for score
            li.score = self.beta * li.score + (1.0 - self.beta) * score
            li.last_seen_step = global_step
            li.visits += 1
        else:
            # Add new level
            self._data[level_id] = LevelInfo(
                level_id=level_id,
                score=score,
                last_seen_step=global_step,
                first_seen_step=global_step,
                visits=1
            )
            self._evict_if_needed(global_step)

    def staleness(self, level_id: int, global_step: int) -> int:
        """Calculate staleness (steps since last seen)."""
        if level_id not in self._data:
            return 0
        li = self._data[level_id]
        return max(0, global_step - li.last_seen_step)

    def _evict_if_needed(self, global_step: int) -> None:
        """
        Evict levels when buffer is full.
        Uses different strategies based on eviction_mode.
        """
        if len(self._data) <= self.max_size:
            return

        n_remove = len(self._data) - self.max_size
        items = list(self._data.values())
        
        if self.eviction_mode == "score":
            # Remove lowest score levels
            items_sorted = sorted(items, key=lambda x: x.score)
            
        elif self.eviction_mode == "lru":
            # Remove least recently used (highest staleness)
            items_sorted = sorted(items, key=lambda x: global_step - x.last_seen_step, reverse=True)
            
        elif self.eviction_mode == "score_staleness":
            # Combined: evict levels with low score AND high staleness
            # Compute combined metric: prioritize for eviction if low score and stale
            def eviction_score(li: LevelInfo) -> float:
                staleness = global_step - li.last_seen_step
                # Normalize both metrics to [0, 1]
                max_staleness = max(1, global_step - min(x.first_seen_step for x in items))
                max_score = max(x.score for x in items) + 1e-6
                
                norm_staleness = staleness / max_staleness
                norm_score = li.score / max_score
                
                # Lower is worse (will be evicted)
                # Low score = bad, high staleness = bad
                return norm_score * 0.7 - norm_staleness * 0.3
            
            items_sorted = sorted(items, key=eviction_score)
            
        elif self.eviction_mode == "random":
            # Random eviction
            self.rng.shuffle(items)
            items_sorted = items
            
        else:
            # Default: score-based
            items_sorted = sorted(items, key=lambda x: x.score)
        
        # Evict worst levels
        for i in range(n_remove):
            del self._data[items_sorted[i].level_id]
    
    def get_statistics(self, global_step: int) -> dict:
        """Return buffer statistics for logging."""
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