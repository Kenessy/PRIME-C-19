import pytest

from prime_c19.tp6.controls import CadenceGovernor


def test_cadence_governor_warmup_returns_start_tau():
    gov = CadenceGovernor(
        start_tau=5.0,
        warmup_steps=2,
        min_tau=1,
        max_tau=10,
        ema=0.5,
        target_flip=0.1,
        grad_high=10.0,
        grad_low=0.1,
        loss_flat=0.001,
        loss_spike=0.5,
        step_up=1.0,
        step_down=1.0,
        vel_high=0.5,
    )

    assert gov.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0) == 5
    assert gov.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0) == 5


def test_cadence_governor_velocity_and_grad_short_circuit():
    gov = CadenceGovernor(
        start_tau=5.0,
        warmup_steps=0,
        min_tau=1,
        max_tau=10,
        ema=0.5,
        target_flip=0.1,
        grad_high=2.0,
        grad_low=0.1,
        loss_flat=0.001,
        loss_spike=0.5,
        step_up=1.0,
        step_down=1.0,
        vel_high=0.5,
    )

    # Velocity high => force min_tau.
    assert gov.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=1.0) == 1

    # Grad norm high => force max_tau.
    assert gov.update(loss_value=1.0, grad_norm=100.0, flip_rate=0.0, ptr_velocity=0.0) == 10


def test_cadence_governor_increases_on_high_flip_and_decreases_on_laminar():
    gov = CadenceGovernor(
        start_tau=5.0,
        warmup_steps=0,
        min_tau=1,
        max_tau=10,
        ema=0.0,  # make EMA fully follow the current value for deterministic tests
        target_flip=0.1,
        grad_high=10.0,
        grad_low=0.1,
        loss_flat=0.001,
        loss_spike=0.5,
        step_up=1.0,
        step_down=1.0,
        vel_high=0.5,
    )

    # High flip => step_up.
    tau1 = gov.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.9, ptr_velocity=0.0)
    assert tau1 == 6

    # Create a new governor to test laminar speed-up without history.
    gov2 = CadenceGovernor(
        start_tau=5.0,
        warmup_steps=0,
        min_tau=1,
        max_tau=10,
        ema=0.0,
        target_flip=0.2,
        grad_high=10.0,
        grad_low=0.1,
        loss_flat=0.001,
        loss_spike=0.5,
        step_up=1.0,
        step_down=1.0,
        vel_high=0.5,
    )

    # First call seeds prev_loss and EMAs.
    gov2.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0)
    # Second call with same loss => abs(loss_delta)=0, laminar => step_down.
    tau2 = gov2.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0)
    assert tau2 == 3
