import os

import pytest

from prime_c19.tp6.controls import ThermostatParams, apply_thermostat


class DummyModel:
    def __init__(self, ptr_inertia=0.5, ptr_deadzone=0.25, ptr_walk_prob=0.1):
        self.ptr_inertia = ptr_inertia
        self.ptr_deadzone = ptr_deadzone
        self.ptr_walk_prob = ptr_walk_prob


def test_apply_thermostat_discrete_steps_adjust_values():
    params = ThermostatParams(
        ema_beta=0.0,
        target_flip=0.1,
        inertia_step=0.1,
        deadzone_step=0.05,
        walk_step=0.02,
        inertia_min=0.0,
        inertia_max=1.0,
        deadzone_min=0.0,
        deadzone_max=1.0,
        walk_min=0.0,
        walk_max=0.5,
    )

    m = DummyModel(ptr_inertia=0.5, ptr_deadzone=0.25, ptr_walk_prob=0.1)

    # High flip -> increase inertia & deadzone, decrease walk
    ema = apply_thermostat(m, flip_rate=0.9, ema=None, params=params)
    assert ema == 0.9
    assert m.ptr_inertia == pytest.approx(0.6)
    assert m.ptr_deadzone == pytest.approx(0.3)
    assert m.ptr_walk_prob == pytest.approx(0.08)

    # Very low flip -> decrease inertia/deadzone, increase walk
    ema = apply_thermostat(m, flip_rate=0.0, ema=ema, params=params)
    assert m.ptr_inertia == pytest.approx(0.5)
    assert m.ptr_deadzone == pytest.approx(0.25)
    assert m.ptr_walk_prob == pytest.approx(0.1)


def test_apply_thermostat_continuous_focus_tension_targets():
    params = ThermostatParams(
        ema_beta=0.9,
        target_flip=0.1,
        inertia_step=0.1,
        deadzone_step=0.05,
        walk_step=0.02,
        inertia_min=0.0,
        inertia_max=1.0,
        deadzone_min=0.0,
        deadzone_max=1.0,
        walk_min=0.0,
        walk_max=0.5,
    )

    m = DummyModel(ptr_inertia=0.0, ptr_deadzone=1.0, ptr_walk_prob=0.5)

    # focus=1, tension=0 => target_inertia=1, target_deadzone=0, target_walk=walk_min.
    ema = apply_thermostat(m, flip_rate=0.0, ema=0.0, params=params, focus=1.0, tension=0.0)

    blend = max(1e-3, 1.0 - params.ema_beta)
    # Moves a small step toward targets (EMA-blend), not jump.
    assert m.ptr_inertia == pytest.approx(0.0 + (1.0 - 0.0) * blend)
    assert m.ptr_deadzone == pytest.approx(1.0 + (0.0 - 1.0) * blend)
    assert m.ptr_walk_prob == pytest.approx(0.5 + (0.0 - 0.5) * blend)
    assert ema == pytest.approx(params.ema_beta * 0.0 + (1.0 - params.ema_beta) * 0.0)


def test_apply_thermostat_respects_env_inertia_override(monkeypatch):
    params = ThermostatParams(
        ema_beta=0.0,
        target_flip=0.1,
        inertia_step=0.1,
        deadzone_step=0.05,
        walk_step=0.02,
        inertia_min=0.0,
        inertia_max=1.0,
        deadzone_min=0.0,
        deadzone_max=1.0,
        walk_min=0.0,
        walk_max=0.5,
    )

    monkeypatch.setenv("TP6_PTR_INERTIA_OVERRIDE", "0.33")
    m = DummyModel(ptr_inertia=0.5, ptr_deadzone=0.25, ptr_walk_prob=0.1)

    ema = apply_thermostat(m, flip_rate=0.9, ema=None, params=params)

    # No mutations when override present.
    assert m.ptr_inertia == pytest.approx(0.5)
    assert m.ptr_deadzone == pytest.approx(0.25)
    assert m.ptr_walk_prob == pytest.approx(0.1)
    assert ema == 0.9
