"""TP6 control loops (AGC, thermostat, inertia auto, cadence, panic).

Extracted from `tournament_phase6.py` to make the kernel easier to test.

IMPORTANT: These functions are behavior-preserving. They mutate the passed-in
`model` object in-place exactly like the original code.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ThermostatParams:
    """Parameters for `apply_thermostat`.

    Names match their meaning in the original kernel:
    - `ema_beta`: EMA coefficient for flip-rate smoothing.
    - `target_flip`: target flip-rate.
    - `*_step`: discrete step sizes (fallback mode).
    - `*_min`/`*_max`: clamp bounds.
    """

    ema_beta: float
    target_flip: float

    inertia_step: float
    deadzone_step: float
    walk_step: float

    inertia_min: float
    inertia_max: float

    deadzone_min: float
    deadzone_max: float

    walk_min: float
    walk_max: float


def apply_thermostat(
    model,
    flip_rate: float,
    ema: float | None,
    params: ThermostatParams,
    *,
    focus: float | None = None,
    tension: float | None = None,
    raw_delta: float | None = None,
) -> float:
    """Adaptive pointer control: reduce flapping without freezing forever.

    Behavior matches the original implementation in `tournament_phase6.py`.

    Returns:
        The updated EMA of flip_rate.
    """

    if ema is None:
        ema = flip_rate
    else:
        ema = params.ema_beta * ema + (1.0 - params.ema_beta) * flip_rate

    # Respect manual inertia override: do not mutate ptr_inertia/deadzone/walk.
    if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is not None:
        return ema

    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    if focus is not None and tension is not None:
        f = _clamp01(float(focus))
        t = _clamp01(float(tension))
        stuck = 0.0
        if raw_delta is not None:
            try:
                rd = float(raw_delta)
            except (TypeError, ValueError):
                rd = None
            if rd is not None and math.isfinite(rd):
                stuck = 1.0 / (1.0 + max(0.0, rd))
        drive = max(t, 1.0 - f, stuck)

        target_inertia = params.inertia_min + (params.inertia_max - params.inertia_min) * (f * (1.0 - t))
        target_deadzone = params.deadzone_min + (params.deadzone_max - params.deadzone_min) * t
        target_walk = params.walk_min + (params.walk_max - params.walk_min) * drive

        blend = max(1e-3, 1.0 - params.ema_beta)
        model.ptr_inertia = model.ptr_inertia + (target_inertia - model.ptr_inertia) * blend
        model.ptr_deadzone = model.ptr_deadzone + (target_deadzone - model.ptr_deadzone) * blend
        model.ptr_walk_prob = model.ptr_walk_prob + (target_walk - model.ptr_walk_prob) * blend
        return ema

    if ema > params.target_flip:
        model.ptr_inertia = min(params.inertia_max, model.ptr_inertia + params.inertia_step)
        model.ptr_deadzone = min(params.deadzone_max, model.ptr_deadzone + params.deadzone_step)
        model.ptr_walk_prob = max(params.walk_min, model.ptr_walk_prob - params.walk_step)
    elif ema < params.target_flip * 0.5:
        model.ptr_inertia = max(params.inertia_min, model.ptr_inertia - params.inertia_step)
        model.ptr_deadzone = max(params.deadzone_min, model.ptr_deadzone - params.deadzone_step)
        model.ptr_walk_prob = min(params.walk_max, model.ptr_walk_prob + params.walk_step)

    return ema


@dataclass(frozen=True)
class AGCParams:
    """Parameters for `apply_update_agc`."""

    enabled: bool

    grad_low: float
    grad_high: float

    scale_up: float
    scale_down: float

    scale_min: float
    scale_max_default: float

    warmup_steps: int
    warmup_init: float


def apply_update_agc(
    model,
    grad_norm: float | None,
    params: AGCParams,
    *,
    raw_delta: float | None = None,
    step: int | None = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> float:
    """Auto-gain control for the state update scale.

    Behavior-preserving extraction of `tournament_phase6.apply_update_agc`.

    The function mutates:
      - `model.update_scale`
      - `model.agc_scale_cap`
      - `model.debug_scale_out`

    Args:
        model: object with `update_scale` attribute (and optionally `agc_scale_max`, `agc_scale_cap`).
        grad_norm: gradient norm signal.
        params: AGC parameters.
        raw_delta: kept for API parity (not used).
        step: global optimizer step.
        log_fn: optional logger callback.

    Returns:
        The new `update_scale`.
    """

    base_cap = float(getattr(model, "agc_scale_max", params.scale_max_default))
    cap = float(getattr(model, "agc_scale_cap", base_cap))
    if not math.isfinite(cap) or cap <= 0:
        cap = base_cap
    cap = max(params.scale_min, min(base_cap, cap))

    # Warmup floor: ramp linearly to scale_min over warmup_steps from warmup_init.
    if step is not None:
        warmup_horizon = max(1, int(params.warmup_steps))
        warmup = max(0.0, min(1.0, step / float(warmup_horizon)))
        floor = params.warmup_init + (params.scale_min - params.warmup_init) * warmup
        floor = max(0.0, min(params.scale_min, floor))
    else:
        floor = params.scale_min

    scale = float(getattr(model, "update_scale", floor))
    if not math.isfinite(scale) or scale <= 0:
        scale = floor
    if step is not None and step == 0:
        scale = floor

    if params.enabled and grad_norm is not None and math.isfinite(float(grad_norm)):
        if grad_norm < params.grad_low:
            scale *= params.scale_up
        elif grad_norm > params.grad_high:
            scale *= params.scale_down

    scale = max(floor, min(cap, scale))
    model.agc_scale_cap = cap
    model.update_scale = scale
    model.debug_scale_out = scale

    if step is not None and step == 0 and log_fn is not None:
        dbg = {
            "scale_in": scale,
            "scale_out": scale,
            "agc_scale_min": params.scale_min,
            "warmup_floor": floor,
            "cap": cap,
            "base_cap": base_cap,
        }
        log_fn(f"[debug_scale_step0] {dbg}")

    return scale


@dataclass(frozen=True)
class InertiaAutoParams:
    """Parameters for `apply_inertia_auto`."""

    enabled: bool

    inertia_min: float
    inertia_max: float

    vel_full: float
    ema_beta: float

    dwell_enabled: bool
    dwell_thresh: float


def apply_inertia_auto(model, ptr_velocity, params: InertiaAutoParams, *, panic_active: bool = False) -> None:
    """Update `model.ptr_inertia` using either dwell or pointer velocity signals."""

    if not params.enabled or panic_active:
        return

    # Dwell-driven kinetic tempering: glue when dwell is high, agile when low.
    if params.dwell_enabled:
        dwell = getattr(model, "ptr_mean_dwell", None)
        max_dwell = getattr(model, "ptr_max_dwell", dwell)
        try:
            dwell = float(dwell) if dwell is not None else 0.0
            max_dwell = float(max_dwell) if max_dwell is not None else dwell
        except (TypeError, ValueError):
            dwell, max_dwell = 0.0, 0.0
        dwell_metric = max(dwell, max_dwell)
        if params.dwell_thresh > 0:
            weight = max(0.0, min(1.0, dwell_metric / params.dwell_thresh))
            target = params.inertia_min + weight * (params.inertia_max - params.inertia_min)
        else:
            target = params.inertia_max
    else:
        if ptr_velocity is None:
            return
        if params.vel_full <= 0:
            return
        try:
            velocity = float(ptr_velocity)
        except (TypeError, ValueError):
            return
        velocity = max(0.0, velocity)
        ratio = min(1.0, velocity / params.vel_full)
        target = params.inertia_min + ratio * (params.inertia_max - params.inertia_min)

    ema = float(getattr(model, "ptr_inertia_ema", model.ptr_inertia))
    ema = params.ema_beta * ema + (1.0 - params.ema_beta) * target
    ema = max(params.inertia_min, min(params.inertia_max, ema))
    model.ptr_inertia_ema = ema
    model.ptr_inertia = ema


class PanicReflex:
    """Loss-based unlock: reduce friction when loss spikes."""

    def __init__(
        self,
        ema_beta: float = 0.9,
        panic_threshold: float = 1.5,
        recovery_rate: float = 0.01,
        inertia_low: float = 0.1,
        inertia_high: float = 0.95,
        walk_prob_max: float = 0.2,
    ) -> None:
        self.loss_ema: float | None = None
        self.beta = ema_beta
        self.threshold = panic_threshold
        self.recovery = recovery_rate
        self.inertia_low = inertia_low
        self.inertia_high = inertia_high
        self.walk_prob_max = walk_prob_max
        self.panic_state = 0.0

    def update(self, loss_value: float) -> dict:
        if self.loss_ema is None:
            self.loss_ema = loss_value
            return {"status": "INIT", "inertia": self.inertia_high, "walk_prob": 0.0}
        ratio = loss_value / (self.loss_ema + 1e-6)
        if ratio > self.threshold:
            self.panic_state = 1.0
        else:
            self.panic_state = max(0.0, self.panic_state - self.recovery)
        self.loss_ema = (self.beta * self.loss_ema) + ((1.0 - self.beta) * loss_value)
        if self.panic_state > 0.1:
            inertia = self.inertia_low + (self.inertia_high - self.inertia_low) * (1.0 - self.panic_state)
            walk_prob = self.walk_prob_max * self.panic_state
            return {"status": "PANIC", "inertia": inertia, "walk_prob": walk_prob}
        return {"status": "LOCKED", "inertia": self.inertia_high, "walk_prob": 0.0}


class CadenceGovernor:
    """Adaptive cadence controller combining flip-rate and gradient shock signals."""

    def __init__(
        self,
        start_tau: float,
        warmup_steps: int,
        min_tau: int,
        max_tau: int,
        ema: float,
        target_flip: float,
        grad_high: float,
        grad_low: float,
        loss_flat: float,
        loss_spike: float,
        step_up: float,
        step_down: float,
        *,
        vel_high: float,
    ):
        self.tau = float(start_tau)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_tau = max(1, int(min_tau))
        self.max_tau = max(self.min_tau, int(max_tau))
        self.ema = float(ema)
        self.target_flip = float(target_flip)
        self.grad_high = float(grad_high)
        self.grad_low = float(grad_low)
        self.loss_flat = float(loss_flat)
        self.loss_spike = float(loss_spike)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.vel_high = float(vel_high)
        self.step_count = 0
        self.grad_ema = None
        self.flip_ema = None
        self.vel_ema = None
        self.prev_loss = None

    def update(self, loss_value: float, grad_norm: float, flip_rate: float, ptr_velocity=None) -> int:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            return int(round(self.tau))

        if ptr_velocity is not None:
            if self.vel_ema is None:
                self.vel_ema = float(ptr_velocity)
            else:
                self.vel_ema = self.ema * self.vel_ema + (1.0 - self.ema) * float(ptr_velocity)
            # High pointer velocity requires higher sampling (lower cadence).
            if self.vel_ema > self.vel_high:
                self.tau = float(self.min_tau)
                self.prev_loss = loss_value
                return int(round(self.tau))

        if grad_norm > self.grad_high:
            self.tau = float(self.max_tau)
            return int(round(self.tau))

        if self.grad_ema is None:
            self.grad_ema = grad_norm
        else:
            self.grad_ema = self.ema * self.grad_ema + (1.0 - self.ema) * grad_norm

        if self.flip_ema is None:
            self.flip_ema = flip_rate
        else:
            self.flip_ema = self.ema * self.flip_ema + (1.0 - self.ema) * flip_rate

        if self.prev_loss is None:
            loss_delta = 0.0
        else:
            loss_delta = self.prev_loss - loss_value
        self.prev_loss = loss_value

        # Slow down when turbulence is high or loss spikes.
        if self.flip_ema > self.target_flip or self.grad_ema > self.grad_high or loss_delta < -self.loss_spike:
            self.tau = min(self.max_tau, self.tau + self.step_up)
        # Speed up only when laminar and loss is flat.
        elif self.grad_ema < self.grad_low and self.flip_ema < self.target_flip * 0.5 and abs(loss_delta) < self.loss_flat:
            self.tau = max(self.min_tau, self.tau - self.step_down)

        self.tau = max(self.min_tau, min(self.max_tau, self.tau))
        return int(round(self.tau))
