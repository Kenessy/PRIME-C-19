"""TP6 batch sharding helpers.

The original kernel uses a scale-free heuristic named `calculate_adaptive_vasc`
(see `tournament_phase6.py`) to pick shard counts that evenly divide the batch.

VASC acronym expansion is not defined in the codebase (unknown).
"""

from __future__ import annotations

from typing import Tuple


def calculate_adaptive_vasc(
    batch_size: int,
    dwell: float,
    grad_norm: float,
    max_dwell_limit: float,
    ema_grad_norm: float,
    *,
    min_group_ratio: float = 0.02,
) -> Tuple[int, int, float, float, float]:
    """Scale-free VASC: choose shard count from ratio-based cohesion signals.

    Returns:
        (shard_count, group_size, focus, tension, cohesion)

    Notes:
        Behavior matches the original function in `tournament_phase6.py`.
    """

    eps = 1e-8
    batch_size_int = int(batch_size)
    if batch_size_int <= 0:
        return 1, 0, 0.0, 0.0, 0.0

    max_dwell_limit = max(eps, float(max_dwell_limit))
    ema_grad_norm = max(eps, float(ema_grad_norm))

    focus = max(0.0, min(1.0, float(dwell) / max_dwell_limit))
    tension = max(0.0, min(1.0, float(grad_norm) / (ema_grad_norm + eps)))
    cohesion = max(0.0, min(1.0, focus - tension))

    ceiling = float(batch_size_int)
    floor = max(1.0, ceiling * float(min_group_ratio))

    target_group_size = floor + (ceiling - floor) * cohesion
    raw_shards = ceiling / max(eps, target_group_size)

    valid_counts = [c for c in range(1, batch_size_int + 1) if batch_size_int % c == 0]
    shard_count = min(valid_counts, key=lambda c: abs(c - raw_shards)) if valid_counts else 1
    shard_count = max(1, min(shard_count, batch_size_int))
    group_size = batch_size_int // shard_count
    return shard_count, group_size, focus, tension, cohesion
