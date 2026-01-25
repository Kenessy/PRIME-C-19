import pytest

from prime_c19.tp6.sharding import calculate_adaptive_vasc


def test_vasc_returns_divisors_and_extremes():
    # cohesion=1 => 1 shard (full batch)
    shard_count, group_size, focus, tension, cohesion = calculate_adaptive_vasc(
        batch_size=32,
        dwell=10.0,
        grad_norm=0.0,
        max_dwell_limit=10.0,
        ema_grad_norm=1.0,
        min_group_ratio=0.02,
    )
    assert (shard_count, group_size) == (1, 32)
    assert focus == pytest.approx(1.0)
    assert tension == pytest.approx(0.0)
    assert cohesion == pytest.approx(1.0)

    # cohesion~0 => target_group_size=floor => more shards.
    shard_count, group_size, focus, tension, cohesion = calculate_adaptive_vasc(
        batch_size=32,
        dwell=0.0,
        grad_norm=1.0,
        max_dwell_limit=10.0,
        ema_grad_norm=1.0,
        min_group_ratio=0.25,  # floor=8 -> raw_shards=4
    )
    assert (shard_count, group_size) == (4, 8)
    assert group_size * shard_count == 32
    assert 0.0 <= focus <= 1.0
    assert 0.0 <= tension <= 1.0
    assert 0.0 <= cohesion <= 1.0


def test_vasc_batch_size_zero_edge():
    shard_count, group_size, focus, tension, cohesion = calculate_adaptive_vasc(
        batch_size=0,
        dwell=0.0,
        grad_norm=0.0,
        max_dwell_limit=1.0,
        ema_grad_norm=1.0,
        min_group_ratio=0.02,
    )
    assert shard_count == 1
    assert group_size == 0
    assert focus == 0.0
    assert tension == 0.0
    assert cohesion == 0.0
