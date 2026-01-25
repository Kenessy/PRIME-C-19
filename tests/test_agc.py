import math

import pytest

from prime_c19.tp6.controls import AGCParams, apply_update_agc


class DummyModel:
    def __init__(self, update_scale=1.0, agc_scale_max=1.0, agc_scale_cap=1.0):
        self.update_scale = update_scale
        self.agc_scale_max = agc_scale_max
        self.agc_scale_cap = agc_scale_cap
        self.debug_scale_out = None


def test_apply_update_agc_warmup_floor_and_logging():
    logs = []

    def log_fn(msg: str) -> None:
        logs.append(msg)

    params = AGCParams(
        enabled=True,
        grad_low=0.5,
        grad_high=2.0,
        scale_up=2.0,
        scale_down=0.5,
        scale_min=0.01,
        scale_max_default=1.0,
        warmup_steps=10,
        warmup_init=0.001,
    )

    m = DummyModel(update_scale=123.0, agc_scale_max=1.0, agc_scale_cap=1.0)

    # step=0 forces scale to the warmup floor first, then applies AGC scaling.
    scale = apply_update_agc(m, grad_norm=0.0, params=params, step=0, log_fn=log_fn)
    assert math.isclose(scale, 0.002, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(m.update_scale, 0.002, rel_tol=0, abs_tol=1e-12)
    assert m.debug_scale_out == m.update_scale
    assert any("debug_scale_step0" in s for s in logs)


def test_apply_update_agc_scales_up_and_down_with_clamps():
    params = AGCParams(
        enabled=True,
        grad_low=1.0,
        grad_high=3.0,
        scale_up=2.0,
        scale_down=0.25,
        scale_min=0.01,
        scale_max_default=0.5,
        warmup_steps=0,
        warmup_init=0.01,
    )

    m = DummyModel(update_scale=0.1, agc_scale_max=0.5, agc_scale_cap=0.5)

    # Low grad -> scale up
    s1 = apply_update_agc(m, grad_norm=0.5, params=params, step=1)
    assert math.isclose(s1, 0.2, rel_tol=0, abs_tol=1e-12)

    # High grad -> scale down
    s2 = apply_update_agc(m, grad_norm=10.0, params=params, step=2)
    assert math.isclose(s2, 0.05, rel_tol=0, abs_tol=1e-12)

    # Clamp to min floor
    m.update_scale = 1e-9
    s3 = apply_update_agc(m, grad_norm=10.0, params=params, step=3)
    assert s3 >= params.scale_min

    # Clamp to cap
    m.update_scale = 100.0
    s4 = apply_update_agc(m, grad_norm=0.0, params=params, step=4)
    assert s4 <= m.agc_scale_cap


def test_apply_update_agc_handles_nonfinite_cap():
    params = AGCParams(
        enabled=True,
        grad_low=0.5,
        grad_high=2.0,
        scale_up=2.0,
        scale_down=0.5,
        scale_min=0.01,
        scale_max_default=1.0,
        warmup_steps=0,
        warmup_init=0.01,
    )

    m = DummyModel(update_scale=0.1, agc_scale_max=1.0, agc_scale_cap=float("nan"))
    s = apply_update_agc(m, grad_norm=0.0, params=params, step=1)
    # Falls back to base cap (agc_scale_max)
    assert s <= 1.0
