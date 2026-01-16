import os
import time
import math
import json
import random
import shutil
import urllib.request
import zipfile
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None
    HAS_TORCHAUDIO = False

ROOT = r"G:\AI\pilot_pulse"
DATA_DIR = os.path.join(ROOT, "data")
LOG_PATH = os.path.join(ROOT, "logs", "current", "tournament_phase6.log")

SEED = int(os.environ.get("TP6_SEED", 123))
DEVICE = os.environ.get("TP6_DEVICE", "").strip().lower()
if DEVICE not in {"cuda", "cpu"}:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFLINE_ONLY = os.environ.get("PILOT_OFFLINE", "1") == "1"

AUDIO_BACKEND = os.environ.get("TP6_AUDIO_BACKEND", "").strip()
if HAS_TORCHAUDIO and AUDIO_BACKEND:
    try:
        torchaudio.set_audio_backend(AUDIO_BACKEND)
    except Exception:
        pass

# experiment budget (env overrides for quick sanity)
MAX_SAMPLES = int(os.environ.get("TP6_MAX_SAMPLES", 5000))
EVAL_SAMPLES = int(os.environ.get("TP6_EVAL_SAMPLES", 1024))
EVAL_SPLIT = os.environ.get("TP6_EVAL_SPLIT", "test").strip().lower()
BATCH_SIZE = int(os.environ.get("TP6_BATCH_SIZE", 128))
LR = float(os.environ.get("TP6_LR", 1e-3))
WALL_CLOCK_SECONDS = int(os.environ.get("TP6_WALL", 15 * 60))  # seconds
MAX_STEPS = int(os.environ.get("TP6_MAX_STEPS", 0))
HEARTBEAT_STEPS = int(os.environ.get("TP6_HEARTBEAT", 10))
HEARTBEAT_SECS = float(os.environ.get("TP6_HEARTBEAT_SECS", 0.0))
LIVE_TRACE_EVERY = int(os.environ.get("TP6_LIVE_TRACE_EVERY", HEARTBEAT_STEPS))
SATIETY_THRESH = float(os.environ.get("TP6_SATIETY", 0.98))
RING_LEN = int(os.environ.get("TP6_RING_LEN", 4096))
PTR_PARAM_STRIDE = int(os.environ.get("TP6_PTR_STRIDE", 1))
# Gaussian window + movement penalty controls
GAUSS_K = int(os.environ.get("TP6_GAUSS_K", 2))  # neighbors on each side; window size = 2*K+1
GAUSS_TAU = float(os.environ.get("TP6_GAUSS_TAU", 0.5))
PTR_KERNEL = os.environ.get("TP6_PTR_KERNEL", "gauss").strip().lower()
PTR_KAPPA = float(os.environ.get("TP6_PTR_KAPPA", 4.0))
PTR_EDGE_EPS = float(os.environ.get("TP6_PTR_EDGE_EPS", 0.0))
LAMBDA_MOVE = float(os.environ.get("TP6_LMOVE", 1e-3))
PTR_INERTIA = float(os.environ.get("TP6_PTR_INERTIA", 0.0))  # 0=no inertia, 0.9=strong stay-bias
PTR_DEADZONE = float(os.environ.get("TP6_PTR_DEADZONE", 0.0))  # distance below which pointer resists moving
PTR_DEADZONE_TAU = float(os.environ.get("TP6_PTR_DEADZONE_TAU", 1e-3))
PTR_WARMUP_STEPS = int(os.environ.get("TP6_PTR_WARMUP_STEPS", 0))
PTR_WALK_PROB = float(os.environ.get("TP6_PTR_WALK_PROB", 0.2))  # 0=stay when not jumping, 1=always walk
PTR_NO_ROUND = os.environ.get("TP6_PTR_NO_ROUND", "0").strip() == "1"
PTR_PHANTOM = os.environ.get("TP6_PTR_PHANTOM", "0").strip() == "1"
PTR_PHANTOM_OFF = float(os.environ.get("TP6_PTR_PHANTOM_OFF", 0.5))
PTR_PHANTOM_READ = os.environ.get("TP6_PTR_PHANTOM_READ", "0").strip() == "1"
SOFT_READOUT = os.environ.get("TP6_SOFT_READOUT", "0").strip() == "1"
SOFT_READOUT_K = int(os.environ.get("TP6_SOFT_READOUT_K", 2))
SOFT_READOUT_TAU = float(os.environ.get("TP6_SOFT_READOUT_TAU", GAUSS_TAU))
PTR_VEL = os.environ.get("TP6_PTR_VEL", "0").strip() == "1"
PTR_VEL_DECAY = float(os.environ.get("TP6_PTR_VEL_DECAY", 0.9))
PTR_VEL_CAP = float(os.environ.get("TP6_PTR_VEL_CAP", 0.5))
PTR_VEL_SCALE = float(os.environ.get("TP6_PTR_VEL_SCALE", 1.0))
PTR_LOCK = os.environ.get("TP6_PTR_LOCK", "0").strip() == "1"
PTR_LOCK_VALUE = float(os.environ.get("TP6_PTR_LOCK_VALUE", 0.5))
PTR_UPDATE_EVERY = int(os.environ.get("TP6_PTR_UPDATE_EVERY", 1))
PTR_UPDATE_AUTO = os.environ.get("TP6_PTR_UPDATE_AUTO", "0").strip() == "1"
PTR_UPDATE_MIN = int(os.environ.get("TP6_PTR_UPDATE_MIN", 1))
PTR_UPDATE_MAX = int(os.environ.get("TP6_PTR_UPDATE_MAX", 16))
PTR_UPDATE_EVERY_STEP = int(os.environ.get("TP6_PTR_UPDATE_EVERY_STEP", 20))
PTR_UPDATE_TARGET_FLIP = float(os.environ.get("TP6_PTR_UPDATE_TARGET_FLIP", 0.2))
PTR_UPDATE_EMA = float(os.environ.get("TP6_PTR_UPDATE_EMA", 0.9))
PTR_GATE_MODE = os.environ.get("TP6_PTR_GATE_MODE", "none").strip().lower()
PTR_GATE_STEPS = os.environ.get("TP6_PTR_GATE_STEPS", "").strip()
PTR_SOFT_GATE = os.environ.get("TP6_PTR_SOFT_GATE", "0").strip() == "1"
THERMO_ENABLED = os.environ.get("TP6_THERMO", "0").strip() == "1"
THERMO_EVERY = int(os.environ.get("TP6_THERMO_EVERY", 20))
THERMO_TARGET_FLIP = float(os.environ.get("TP6_THERMO_TARGET_FLIP", 0.2))
THERMO_EMA = float(os.environ.get("TP6_THERMO_EMA", 0.9))
THERMO_INERTIA_STEP = float(os.environ.get("TP6_THERMO_INERTIA_STEP", 0.05))
THERMO_DEADZONE_STEP = float(os.environ.get("TP6_THERMO_DEADZONE_STEP", 0.02))
THERMO_WALK_STEP = float(os.environ.get("TP6_THERMO_WALK_STEP", 0.02))
THERMO_INERTIA_MIN = float(os.environ.get("TP6_THERMO_INERTIA_MIN", 0.0))
THERMO_INERTIA_MAX = float(os.environ.get("TP6_THERMO_INERTIA_MAX", 0.95))
THERMO_DEADZONE_MIN = float(os.environ.get("TP6_THERMO_DEADZONE_MIN", 0.0))
THERMO_DEADZONE_MAX = float(os.environ.get("TP6_THERMO_DEADZONE_MAX", 0.5))
THERMO_WALK_MIN = float(os.environ.get("TP6_THERMO_WALK_MIN", 0.0))
THERMO_WALK_MAX = float(os.environ.get("TP6_THERMO_WALK_MAX", 0.3))

PANIC_ENABLED = os.environ.get("TP6_PANIC", "0").strip() == "1"
PANIC_THRESHOLD = float(os.environ.get("TP6_PANIC_THRESHOLD", 1.5))
PANIC_BETA = float(os.environ.get("TP6_PANIC_BETA", 0.9))
PANIC_RECOVERY = float(os.environ.get("TP6_PANIC_RECOVERY", 0.01))
PANIC_INERTIA_LOW = float(os.environ.get("TP6_PANIC_INERTIA_LOW", 0.1))
PANIC_INERTIA_HIGH = float(os.environ.get("TP6_PANIC_INERTIA_HIGH", 0.95))
PANIC_WALK_MAX = float(os.environ.get("TP6_PANIC_WALK_MAX", 0.2))
MOBIUS_ENABLED = os.environ.get("TP6_MOBIUS", "0").strip() == "1"
MOBIUS_EMB_SCALE = float(os.environ.get("TP6_MOBIUS_EMB", 0.1))
ACT_NAME = os.environ.get("TP6_ACT", "identity").strip().lower()
C13_P = float(os.environ.get("TP6_C13_P", 2.0))
DEBUG_NAN = os.environ.get("TP6_DEBUG_NAN", "0").strip() == "1"
DEBUG_STATS = os.environ.get("TP6_DEBUG_STATS", "0").strip() == "1"
DEBUG_EVERY = int(os.environ.get("TP6_DEBUG_EVERY", 0))
PRECISION = os.environ.get("TP6_PRECISION", "fp32").strip().lower()
DTYPE_MAP = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "amp": torch.float16,
}
DTYPE = DTYPE_MAP.get(PRECISION, torch.float32)
USE_AMP = DEVICE == "cuda" and PRECISION in {"fp16", "bf16", "amp"}
if PRECISION == "fp64":
    USE_AMP = False
torch.set_default_dtype(DTYPE)

if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    def amp_autocast():
        return torch.amp.autocast(device_type="cuda", enabled=USE_AMP)

    def amp_grad_scaler():
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
else:
    def amp_autocast():
        return torch.cuda.amp.autocast(enabled=USE_AMP)

    def amp_grad_scaler():
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)
MI_SHUFFLE = os.environ.get("TP6_MI_SHUFFLE", "0").strip() == "1"
STATE_LOOP_METRICS = os.environ.get("TP6_STATE_LOOP_METRICS", "0").strip() == "1"
STATE_LOOP_EVERY = int(os.environ.get("TP6_STATE_LOOP_EVERY", 1))
STATE_LOOP_SAMPLES = int(os.environ.get("TP6_STATE_LOOP_SAMPLES", 0))
STATE_LOOP_DIM = int(os.environ.get("TP6_STATE_LOOP_DIM", 16))
GRAD_CLIP = float(os.environ.get("TP6_GRAD_CLIP", 0.0))
STATE_CLIP = float(os.environ.get("TP6_STATE_CLIP", 0.0))
STATE_DECAY = float(os.environ.get("TP6_STATE_DECAY", 1.0))
UPDATE_SCALE = float(os.environ.get("TP6_UPDATE_SCALE", 1.0))
LIVE_TRACE_PATH = os.environ.get("TP6_LIVE_TRACE", os.path.join(ROOT, "traces", "current", "live_trace.json"))
RUN_MODE = os.environ.get("TP6_MODE", "train")
# Checkpoint / resume controls
CHECKPOINT_PATH = os.environ.get("TP6_CKPT", os.path.join(ROOT, "checkpoint.pt"))
SAVE_EVERY_STEPS = int(os.environ.get("TP6_SAVE_EVERY", 0))
RESUME = os.environ.get("TP6_RESUME", "0") == "1"
LOSS_KEEP = int(os.environ.get("TP6_LOSS_KEEP", 2000))
# Lockout test controls (synthetic, deterministic)
PHASE_A_STEPS = int(os.environ.get("TP6_PHASE_A_STEPS", 50))
PHASE_B_STEPS = int(os.environ.get("TP6_PHASE_B_STEPS", 50))
SYNTH_LEN = int(os.environ.get("TP6_SYNTH_LEN", 256))
SYNTH_SHUFFLE = os.environ.get("TP6_SYNTH_SHUFFLE", "0") == "1"
SYNTH_MODE = os.environ.get("TP6_SYNTH_MODE", "random").strip().lower()
HAND_MIN = int(os.environ.get("TP6_HAND_MIN", 256))
SYNTH_META = {}
# Evolution defaults (small to keep runs short/safe)
EVO_POP = int(os.environ.get("TP6_EVO_POP", 6))
EVO_GENS = int(os.environ.get("TP6_EVO_GENS", 3))
EVO_STEPS = int(os.environ.get("TP6_EVO_STEPS", 100))
EVO_MUT_STD = float(os.environ.get("TP6_EVO_MUT_STD", 0.02))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def rotate_artifacts() -> None:
    """Move current logs/traces/summaries -> last, and last -> archive/<timestamp>."""
    ts = time.strftime("%Y%m%d_%H%M%S")

    def _rotate_dir(base_dir: str) -> None:
        current_dir = os.path.join(base_dir, "current")
        last_dir = os.path.join(base_dir, "last")
        archive_dir = os.path.join(base_dir, "archive")

        os.makedirs(current_dir, exist_ok=True)
        os.makedirs(last_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)

        if os.path.isdir(last_dir) and os.listdir(last_dir):
            archive_run = os.path.join(archive_dir, ts)
            os.makedirs(archive_run, exist_ok=True)
            for name in os.listdir(last_dir):
                shutil.move(os.path.join(last_dir, name), os.path.join(archive_run, name))

        if os.path.isdir(current_dir) and os.listdir(current_dir):
            for name in os.listdir(current_dir):
                shutil.move(os.path.join(current_dir, name), os.path.join(last_dir, name))

    _rotate_dir(os.path.join(ROOT, "logs"))
    _rotate_dir(os.path.join(ROOT, "traces"))
    _rotate_dir(os.path.join(ROOT, "summaries"))


def sync_current_to_last() -> None:
    """Copy current logs/traces/summaries into logs/last for quick inspection."""
    dest_dir = os.path.join(ROOT, "logs", "last")
    os.makedirs(dest_dir, exist_ok=True)
    for rel in ("logs/current", "traces/current", "summaries/current"):
        src_dir = os.path.join(ROOT, rel)
        if not os.path.isdir(src_dir):
            continue
        for name in os.listdir(src_dir):
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, name))


def nan_guard(name: str, tensor: torch.Tensor, step: int) -> None:
    if not DEBUG_NAN:
        return
    if not tensor.is_floating_point():
        return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        log(f"[nan_guard] step={step:04d} tensor={name} has NaN/Inf")
        raise RuntimeError(f"NaN/Inf in {name} at step {step}")


def apply_thermostat(model, flip_rate: float, ema: float | None):
    """Adaptive pointer control: reduce flapping without freezing forever."""
    if ema is None:
        ema = flip_rate
    else:
        ema = THERMO_EMA * ema + (1.0 - THERMO_EMA) * flip_rate

    if ema > THERMO_TARGET_FLIP:
        model.ptr_inertia = min(THERMO_INERTIA_MAX, model.ptr_inertia + THERMO_INERTIA_STEP)
        model.ptr_deadzone = min(THERMO_DEADZONE_MAX, model.ptr_deadzone + THERMO_DEADZONE_STEP)
        model.ptr_walk_prob = max(THERMO_WALK_MIN, model.ptr_walk_prob - THERMO_WALK_STEP)
    elif ema < THERMO_TARGET_FLIP * 0.5:
        model.ptr_inertia = max(THERMO_INERTIA_MIN, model.ptr_inertia - THERMO_INERTIA_STEP)
        model.ptr_deadzone = max(THERMO_DEADZONE_MIN, model.ptr_deadzone - THERMO_DEADZONE_STEP)
        model.ptr_walk_prob = min(THERMO_WALK_MAX, model.ptr_walk_prob + THERMO_WALK_STEP)

    return ema


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


def compute_slope(losses: List[float]) -> float:
    if len(losses) < 2:
        return float("nan")
    x = np.arange(len(losses), dtype=np.float64)
    y = np.array(losses, dtype=np.float64)
    a, _ = np.polyfit(x, y, 1)
    return float(a)


class AbsoluteHallway(nn.Module):
    """
    Boundaryless ring with intrinsic pointer params per neuron.
    - Each neuron has theta_ptr (target coord) and theta_gate (bias).
    - Pointer update is a soft mix of jump target and walk/stay, then optional inertia/deadzone.
    - Readout: average of states at last K pointers (tensorized) or soft readout window.
    - Satiety exit: if max prob > SATIETY_THRESH, stop processing further timesteps for that sample.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        ring_len=RING_LEN,
        slot_dim=8,
        ptr_stride=PTR_PARAM_STRIDE,
        gauss_k=GAUSS_K,
        gauss_tau=GAUSS_TAU,
    ):
        super().__init__()
        self.ring_len = ring_len
        self.slot_dim = slot_dim
        self.ptr_stride = max(1, ptr_stride)
        self.gauss_k = gauss_k
        self.gauss_tau = gauss_tau
        self.ptr_kernel = PTR_KERNEL if PTR_KERNEL in {"gauss", "vonmises"} else "gauss"
        self.ptr_kappa = PTR_KAPPA
        self.ptr_edge_eps = PTR_EDGE_EPS
        self.act_name = ACT_NAME
        self.c13_p = max(C13_P, 1e-6)
        self.c19_rho = 4.0
        self.c14_rho = self.c19_rho
        self.mobius = MOBIUS_ENABLED
        self.mobius_scale = 2 if self.mobius else 1
        self.ring_range = ring_len * self.mobius_scale
        # Pointer control (modifiable at runtime)
        self.ptr_inertia = PTR_INERTIA
        self.ptr_deadzone = PTR_DEADZONE
        self.ptr_deadzone_tau = PTR_DEADZONE_TAU
        self.ptr_walk_prob = PTR_WALK_PROB
        self.ptr_vel_enabled = PTR_VEL
        self.ptr_vel_decay = PTR_VEL_DECAY
        self.ptr_vel_cap = PTR_VEL_CAP
        self.ptr_vel_scale = PTR_VEL_SCALE
        self.ptr_lock = PTR_LOCK
        self.ptr_lock_value = PTR_LOCK_VALUE
        self.ptr_update_every = max(1, PTR_UPDATE_EVERY)
        self.ptr_update_auto = PTR_UPDATE_AUTO
        self.ptr_update_min = max(1, PTR_UPDATE_MIN)
        self.ptr_update_max = max(self.ptr_update_min, PTR_UPDATE_MAX)
        self.ptr_update_every_step = max(1, PTR_UPDATE_EVERY_STEP)
        self.ptr_update_target_flip = PTR_UPDATE_TARGET_FLIP
        self.ptr_update_ema = PTR_UPDATE_EMA
        self.ptr_update_ema_state = None
        self.ptr_gate_mode = PTR_GATE_MODE
        self.ptr_gate_steps = set()
        if self.ptr_gate_mode == "steps" and PTR_GATE_STEPS:
            for token in PTR_GATE_STEPS.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    self.ptr_gate_steps.add(int(token))
                except ValueError:
                    continue
        self.ptr_soft_gate = PTR_SOFT_GATE
        if self.ptr_soft_gate:
            self.gate_head = nn.Linear(slot_dim, 1)
        else:
            self.gate_head = None
        self.ptr_warmup_steps = PTR_WARMUP_STEPS
        self.ptr_no_round = PTR_NO_ROUND
        self.ptr_phantom = PTR_PHANTOM
        self.ptr_phantom_off = PTR_PHANTOM_OFF
        self.ptr_phantom_read = PTR_PHANTOM_READ
        self.soft_readout = SOFT_READOUT
        self.soft_readout_k = max(0, SOFT_READOUT_K)
        self.soft_readout_tau = max(SOFT_READOUT_TAU, 1e-6)
        self.state_loop_metrics = STATE_LOOP_METRICS
        self.state_loop_every = max(1, STATE_LOOP_EVERY)
        self.state_loop_samples = max(0, STATE_LOOP_SAMPLES)
        self.state_loop_dim = max(2, STATE_LOOP_DIM)
        if self.state_loop_metrics:
            g = torch.Generator()
            g.manual_seed(1337)
            proj = torch.randn(self.slot_dim, self.state_loop_dim, generator=g)
            proj = torch.nn.functional.normalize(proj, dim=0)
            self.register_buffer("state_loop_proj", proj)
        else:
            self.register_buffer("state_loop_proj", torch.empty(0))
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.jump_score = nn.Linear(slot_dim, 1)
        # intrinsic params (downsampled for memory, then upsampled by gather)
        reduced = (self.ring_range + self.ptr_stride - 1) // self.ptr_stride
        self.theta_ptr_reduced = nn.Parameter(torch.zeros(reduced))  # mapped via sigmoid to [0,1]
        self.theta_gate_reduced = nn.Parameter(torch.zeros(reduced))
        if self.mobius:
            self.phase_embed = nn.Parameter(torch.zeros(2, slot_dim))
        else:
            self.register_parameter("phase_embed", None)
        self.head = nn.Linear(slot_dim, num_classes)
        self.pointer_hist_bins = 128
        self.register_buffer("bin_edges", torch.linspace(0, self.ring_range, self.pointer_hist_bins + 1))
        self.pointer_hist = torch.zeros(self.pointer_hist_bins, dtype=torch.long)
        self.satiety_exits = 0
        self.readout_k = 5
        self.debug_stats = None
        self.reset_parameters()

    def reset_parameters(self):
        # Spread ptr targets across ring; bias gates near zero
        nn.init.uniform_(self.theta_ptr_reduced, -4.0, 4.0)
        nn.init.uniform_(self.theta_gate_reduced, -0.5, 0.5)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)
        nn.init.xavier_uniform_(self.jump_score.weight)
        nn.init.zeros_(self.jump_score.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.gate_head is not None:
            nn.init.xavier_uniform_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)
        if self.mobius:
            nn.init.normal_(self.phase_embed, mean=0.0, std=MOBIUS_EMB_SCALE)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name in ("identity", "none"):
            return x
        if self.act_name == "tanh":
            return torch.tanh(x)
        if self.act_name == "softsign":
            return x / (1.0 + x.abs())
        if self.act_name == "arctan":
            return torch.atan(x)
        if self.act_name in ("silu", "swish"):
            return torch.nn.functional.silu(x)
        if self.act_name == "relu":
            return torch.relu(x)
        if self.act_name in ("c13", "c-13"):
            t = 1.0 + (x / self.c13_p)
            t = torch.clamp(t, 0.0, 1.0)
            return x * t * t
        if self.act_name in ("c13-static", "c-13-static"):
            t = 1.0 + (x / 2.0)
            t = torch.clamp(t, 0.0, 1.0)
            return x * t * t
        if self.act_name in ("c19", "c-19", "candidate-19", "c14", "c-14"):
            u = x
            l = 6.0 * math.pi
            inv_pi = 1.0 / math.pi
            scaled = u * inv_pi
            n = torch.floor(scaled)
            t = scaled - n
            h = t * (1.0 - t)
            is_even = torch.remainder(n, 2.0) < 1.0
            sgn = torch.where(is_even, torch.ones_like(u), -torch.ones_like(u))
            core = math.pi * (sgn * h + (self.c19_rho * h * h))
            return torch.where(u >= l, u - l, torch.where(u <= -l, u + l, core))
        return x

    def _gather_params(self, ptr):
        # ptr: [B] float or long, map to reduced indices with linear interpolation.
        ptr_f = ptr.to(torch.float32)
        idx_float = ptr_f / self.ptr_stride
        idx_base = torch.floor(idx_float)
        frac = (idx_float - idx_base).clamp(0.0, 1.0).detach().unsqueeze(1)
        n = self.theta_ptr_reduced.numel()
        idx0 = torch.remainder(idx_base, n).long()
        idx1 = torch.remainder(idx0 + 1, n).long()
        ring_range = self.ring_range

        theta_ptr0 = torch.sigmoid(self.theta_ptr_reduced[idx0]) * (ring_range - 1)
        theta_ptr1 = torch.sigmoid(self.theta_ptr_reduced[idx1]) * (ring_range - 1)
        theta_ptr = theta_ptr0 + (theta_ptr1 - theta_ptr0) * frac.squeeze(1)

        theta_gate0 = self.theta_gate_reduced[idx0]
        theta_gate1 = self.theta_gate_reduced[idx1]
        theta_gate = theta_gate0 + (theta_gate1 - theta_gate0) * frac.squeeze(1)
        return theta_ptr, theta_gate

    def _compute_kernel_weights(self, ptr_float, offsets, ring_range, tau_override=None):
        pos = ptr_float.unsqueeze(1) + offsets.unsqueeze(0)
        pos_mod = torch.remainder(pos, ring_range)
        pos_mod = torch.nan_to_num(pos_mod, nan=0.0, posinf=ring_range - 1, neginf=0.0)
        pos_idx = pos_mod.clamp(0, ring_range - 1).long()
        pos_centers = pos_idx.to(ptr_float.dtype)
        if self.ptr_kernel == "vonmises":
            angle_scale = (2.0 * math.pi) / max(ring_range, 1e-6)
            delta = (pos_centers - ptr_float.unsqueeze(1)) * angle_scale
            kappa = max(self.ptr_kappa, 1e-6)
            logits = kappa * torch.cos(delta)
        else:
            delta = torch.remainder(pos_centers - ptr_float.unsqueeze(1) + ring_range / 2, ring_range) - ring_range / 2
            d2 = delta ** 2
            tau = max(self.gauss_tau if tau_override is None else tau_override, 1e-4)
            logits = -d2 / tau
        weights = torch.softmax(logits, dim=1)
        return pos_idx, weights, pos_mod

    def set_ptr_controls(self, inertia=None, deadzone=None, walk_prob=None):
        if inertia is not None:
            self.ptr_inertia = float(inertia)
        if deadzone is not None:
            self.ptr_deadzone = float(deadzone)
        if walk_prob is not None:
            self.ptr_walk_prob = float(walk_prob)

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        ring_range = self.ring_range
        ptr_dtype = torch.float32
        state = torch.zeros(B, ring_range, self.slot_dim, device=device, dtype=x.dtype)
        # randomize start pointer per sample to break symmetry (float for STE)
        if self.ptr_lock:
            ptr_float = torch.full((B,), self.ptr_lock_value, device=device, dtype=ptr_dtype) * (ring_range - 1)
        else:
            ptr_float = torch.rand(B, device=device, dtype=ptr_dtype) * (ring_range - 1)
        # last K pointers tensorized (initialize from starting pointer)
        ptr_int_init = torch.floor(torch.remainder(ptr_float, ring_range)).clamp(0, ring_range - 1).long()
        last_ptrs = ptr_int_init.view(B, 1).repeat(1, self.readout_k)
        hist = torch.zeros(self.pointer_hist_bins, device=device, dtype=torch.long)
        satiety_exited = torch.zeros(B, device=device, dtype=torch.bool)
        ptr_vel = torch.zeros(B, device=device, dtype=ptr_dtype)

        def circ_delta(a, b):
            # Shortest signed delta from a to b on a ring.
            return torch.remainder(b - a + ring_range / 2, ring_range) - ring_range / 2

        def circ_lerp(a, b, w):
            # Move from a toward b by fraction w along shortest arc.
            return torch.remainder(a + w * circ_delta(a, b), ring_range)

        movement_cost = 0.0
        # Dynamic pointer trace (loops/motion)
        prev_ptr_int = None
        prev_prev_ptr_int = None
        dwell_len = torch.zeros(B, device=device, dtype=torch.long)
        max_dwell = torch.zeros(B, device=device, dtype=torch.long)
        flip_count = torch.zeros(B, device=device, dtype=torch.long)
        pingpong_count = torch.zeros(B, device=device, dtype=torch.long)
        total_active_steps = 0
        active_steps_per_sample = torch.zeros(B, device=device, dtype=torch.long)
        # Internal state loop metrics (A-B-A patterns in hidden state modes)
        if self.state_loop_metrics:
            loop_samples = B if self.state_loop_samples <= 0 else min(B, self.state_loop_samples)
            mode_prev = torch.full((loop_samples,), -1, device=device, dtype=torch.long)
            mode_prevprev = torch.full((loop_samples,), -1, device=device, dtype=torch.long)
            mode_dwell = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_max_dwell = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_flip = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_abab = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_counts = torch.zeros(self.state_loop_dim, device=device, dtype=torch.long)
            mode_steps = 0

        for t in range(T):
            active_mask = ~satiety_exited
            if not active_mask.any():
                break

            # Hard guard against NaN/Inf pointer values before any indexing.
            ptr_float = torch.nan_to_num(ptr_float, nan=0.0, posinf=ring_range - 1, neginf=0.0)

            if STATE_DECAY < 1.0:
                decay = min(max(STATE_DECAY, 0.0), 1.0)
                # Freeze inactive samples: apply decay only to active entries.
                decay_vec = active_mask.to(state.dtype) * decay + (~active_mask).to(state.dtype)
                state = state * decay_vec.view(B, 1, 1)

            inp = self.input_proj(x[:, t, :])  # [B, slot_dim]
            inp = self._apply_activation(inp)
            nan_guard("inp", inp, t)
            # Gaussian soft neighborhood over offsets [-K..K]
            offsets = torch.arange(-self.gauss_k, self.gauss_k + 1, device=device, dtype=ptr_float.dtype)
            pos_idx, weights, pos_mod = self._compute_kernel_weights(ptr_float, offsets, ring_range)
            nan_guard("weights", weights, t)
            # gather neighbors
            pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim).clamp(0, ring_range - 1)
            neigh = state.gather(1, pos_idx_exp)  # [B,2K+1,slot_dim]
            cur = (weights.unsqueeze(-1) * neigh.to(weights.dtype)).sum(dim=1)
            if self.mobius:
                # Continuous Riemann helix: smooth phase over [0, 2*ring_len]
                # theta = 0..2pi as ptr_float goes 0..2*ring_len
                theta = (ptr_float / self.ring_len) * math.pi
                phase_cos = torch.cos(theta).unsqueeze(1)
                phase_sin = torch.sin(theta).unsqueeze(1)
                cur = cur + phase_cos * self.phase_embed[0] + phase_sin * self.phase_embed[1]

            if cur.dtype != inp.dtype:
                cur = cur.to(inp.dtype)
            upd = self.gru(inp, cur)
            nan_guard("upd", upd, t)
            # satiety_exited: keep updating logits, but do not write state for inactive samples.
            # (Pointer freeze handled later.)
            # State loop metrics (mode sequence)
            if self.state_loop_metrics and (t % self.state_loop_every == 0):
                loop_active = active_mask[:loop_samples]
                proj = upd[:loop_samples] @ self.state_loop_proj.to(device)
                mode = torch.argmax(proj, dim=1)
                mode_counts += torch.bincount(mode, minlength=self.state_loop_dim)
                if mode_prev[0] == -1:
                    mode_prev = mode
                    mode_prevprev = mode
                    mode_dwell = torch.where(loop_active, torch.ones_like(mode_dwell), mode_dwell)
                    mode_max_dwell = torch.maximum(mode_max_dwell, mode_dwell)
                else:
                    mflip = loop_active & (mode != mode_prev)
                    mode_flip = mode_flip + mflip.long()
                    mode_dwell = torch.where(
                        loop_active,
                        torch.where(mflip, torch.ones_like(mode_dwell), mode_dwell + 1),
                        mode_dwell,
                    )
                    mode_max_dwell = torch.maximum(mode_max_dwell, mode_dwell)
                    mabab = loop_active & (mode == mode_prevprev) & (mode != mode_prev)
                    mode_abab = mode_abab + mabab.long()
                    mode_prevprev = mode_prev
                    mode_prev = mode
                mode_steps += int(loop_active.sum().item())
            # scatter-add updates using the same Gaussian weights
            upd_exp = upd.unsqueeze(1).expand(-1, weights.size(1), -1)
            contrib = (weights.unsqueeze(-1) * upd_exp).to(state.dtype)
            if UPDATE_SCALE != 1.0:
                contrib = contrib * UPDATE_SCALE
            contrib = contrib * active_mask.view(B, 1, 1).to(contrib.dtype)
            state = state.scatter_add(1, pos_idx_exp, contrib)
            if STATE_CLIP > 0.0:
                state = state.clamp(-STATE_CLIP, STATE_CLIP)

            prev_ptr = ptr_float
            jump_p = None
            move_mask = None
            gate = None
            update_allowed = (t % self.ptr_update_every) == 0
            if self.ptr_gate_mode == "steps" and self.ptr_gate_steps:
                update_allowed = update_allowed and (t in self.ptr_gate_steps)
            if self.ptr_lock or not update_allowed:
                ptr_float = prev_ptr
            elif self.ptr_warmup_steps > 0 and t < self.ptr_warmup_steps:
                # Warmup lock: keep pointer fixed to build basic features first.
                ptr_float = prev_ptr
            else:
                theta_ptr, theta_gate = self._gather_params(ptr_float)  # base idx for params
                jump_logits = self.jump_score(upd).squeeze(1) + theta_gate
                nan_guard("jump_logits", jump_logits, t)
                p = torch.sigmoid(jump_logits)
                jump_p = p
                # straight-through estimator for pointer target (continuous)
                target_cont = theta_ptr  # already in [0, ring_len)
                if self.ptr_no_round:
                    target_ste = target_cont
                else:
                    target_ste = (target_cont.round() - target_cont).detach() + target_cont
                walk_ptr = torch.remainder(ptr_float + 1, ring_range)
                # allow "stay" when not jumping to reduce flapping
                walk_prob = min(max(self.ptr_walk_prob, 0.0), 1.0)
                stay_ptr = prev_ptr
                non_jump_ptr = circ_lerp(stay_ptr, walk_ptr, walk_prob)
                # soft mix keeps gradients flowing through p and target_ste
                ptr_float = circ_lerp(non_jump_ptr, target_ste, p)
                # optional inertia (stay-bias)
                inertia = min(max(self.ptr_inertia, 0.0), 0.99)
                if inertia > 0.0:
                    ptr_float = circ_lerp(prev_ptr, ptr_float, 1.0 - inertia)
                # optional deadzone with smooth mask (keeps gradients flowing)
                if self.ptr_deadzone > 0.0:
                    delta_raw = circ_delta(prev_ptr, ptr_float)
                    tau = max(self.ptr_deadzone_tau, 1e-6)
                    move_mask = torch.sigmoid((delta_raw.abs() - self.ptr_deadzone) / tau)
                    ptr_float = torch.remainder(prev_ptr + move_mask * delta_raw, ring_range)
                # Optional velocity governor: smooths large pointer jumps into bounded motion.
                if self.ptr_vel_enabled:
                    delta_to_target = circ_delta(prev_ptr, ptr_float)
                    scale = max(self.ptr_vel_scale, 1e-6)
                    torque = torch.tanh(delta_to_target / scale) * self.ptr_vel_cap
                    ptr_vel = self.ptr_vel_decay * ptr_vel + (1.0 - self.ptr_vel_decay) * torque
                    ptr_float = prev_ptr + ptr_vel
                # Optional learned soft gate: modulates how strongly the pointer updates.
                if self.ptr_soft_gate and self.gate_head is not None:
                    gate = torch.sigmoid(self.gate_head(upd)).squeeze(1)
                    delta_gate = circ_delta(prev_ptr, ptr_float)
                    ptr_float = torch.remainder(prev_ptr + gate * delta_gate, ring_range)
            if self.ptr_vel_enabled:
                ptr_vel = torch.where(active_mask, ptr_vel, torch.zeros_like(ptr_vel))
            ptr_float = torch.where(active_mask, ptr_float, prev_ptr)
            # hard clamp for safety (keeps pointer in-bounds)
            ptr_float = torch.nan_to_num(ptr_float, nan=0.0, posinf=ring_range - 1, neginf=0.0)
            ptr_float = torch.remainder(ptr_float, ring_range)
            nan_guard("ptr_float", ptr_float, t)
            # movement cost (wrap-aware)
            delta = torch.remainder(ptr_float - prev_ptr + ring_range / 2, ring_range) - ring_range / 2
            movement_cost = movement_cost + delta.abs().mean()

            # update history tensorized: prepend ptr, drop last
            ptr_float_phys = torch.remainder(ptr_float, ring_range)
            ptr_base = torch.floor(ptr_float_phys)
            if self.ptr_phantom and prev_ptr_int is not None:
                # Dual-grid hysteresis: use an offset quantizer; if they disagree, hold prior bin.
                ptr_off = torch.floor(torch.remainder(ptr_float_phys + self.ptr_phantom_off, ring_range))
                agree = ptr_base == ptr_off
                ptr_int = torch.where(agree, ptr_base, prev_ptr_int.float())
            else:
                ptr_int = ptr_base
            ptr_int = torch.clamp(ptr_int, 0, ring_range - 1).long()
            if self.ptr_phantom_read:
                ptr_float_phys = ptr_int.float()
            if DEBUG_STATS and (DEBUG_EVERY <= 0 or t % DEBUG_EVERY == 0):
                stats = {
                    "ptr_float_min": float(ptr_float.min().item()),
                    "ptr_float_max": float(ptr_float.max().item()),
                    "ptr_delta_abs_mean": float(delta.abs().mean().item()),
                    "ptr_delta_abs_max": float(delta.abs().max().item()),
                    "ptr_int_unique": int(ptr_int.unique().numel()),
                    "cur_abs_max": float(cur.abs().max().item()),
                    "upd_abs_max": float(upd.abs().max().item()),
                }
                if jump_p is not None:
                    stats["jump_p_mean"] = float(jump_p.mean().item())
                    stats["jump_p_min"] = float(jump_p.min().item())
                    stats["jump_p_max"] = float(jump_p.max().item())
                if move_mask is not None:
                    stats["move_mask_mean"] = float(move_mask.mean().item())
                if gate is not None:
                    stats["gate_mean"] = float(gate.mean().item())
                if self.ptr_vel_enabled:
                    stats["ptr_vel_abs_mean"] = float(ptr_vel.abs().mean().item())
                    stats["ptr_vel_abs_max"] = float(ptr_vel.abs().max().item())
                stats["ptr_update_every"] = int(self.ptr_update_every)
                stats["ptr_soft_gate"] = int(self.ptr_soft_gate)
                stats["ptr_vel_enabled"] = int(self.ptr_vel_enabled)
                if self.ptr_edge_eps > 0.0:
                    eps = self.ptr_edge_eps
                    edge_mask = (ptr_float_phys < eps) | (ptr_float_phys > (ring_range - eps))
                    stats["ptr_edge_rate"] = float(edge_mask.float().mean().item())
                stats["ptr_kernel"] = self.ptr_kernel
                self.debug_stats = stats
            last_ptrs = torch.cat([ptr_int.view(B, 1), last_ptrs[:, :-1]], dim=1)
            bins = torch.bucketize(ptr_int.float(), self.bin_edges.to(device)) - 1
            bins = bins.clamp(0, self.pointer_hist_bins - 1)
            # Count all samples this step; bincount avoids accidental batch collapse
            step_counts = torch.bincount(bins, minlength=self.pointer_hist_bins)
            hist = hist + step_counts

            # Dynamic pointer trace metrics (only count active samples)
            if prev_ptr_int is None:
                prev_ptr_int = ptr_int
                prev_prev_ptr_int = ptr_int
                dwell_len = torch.where(active_mask, torch.ones_like(dwell_len), dwell_len)
                max_dwell = torch.maximum(max_dwell, dwell_len)
            else:
                flip = active_mask & (ptr_int != prev_ptr_int)
                flip_count = flip_count + flip.long()
                # Dwell length updates only for active samples
                dwell_len = torch.where(
                    active_mask,
                    torch.where(flip, torch.ones_like(dwell_len), dwell_len + 1),
                    dwell_len,
                )
                max_dwell = torch.maximum(max_dwell, dwell_len)
                pingpong = active_mask & (ptr_int == prev_prev_ptr_int) & (ptr_int != prev_ptr_int)
                pingpong_count = pingpong_count + pingpong.long()
                prev_prev_ptr_int = prev_ptr_int
                prev_ptr_int = ptr_int
            total_active_steps += int(active_mask.sum().item())
            active_steps_per_sample += active_mask.long()

            # Optional auto-adjust of pointer update cadence (simple EMA controller)
            if self.ptr_update_auto and (t % self.ptr_update_every_step == 0) and total_active_steps > 0:
                flip_rate = float(flip_count.sum().item() / max(1, total_active_steps))
                if self.ptr_update_ema_state is None:
                    ema = flip_rate
                else:
                    ema = self.ptr_update_ema * self.ptr_update_ema_state + (1.0 - self.ptr_update_ema) * flip_rate
                self.ptr_update_ema_state = ema
                # If flapping is high, slow down pointer updates (larger stride)
                if ema > self.ptr_update_target_flip:
                    self.ptr_update_every = min(self.ptr_update_max, self.ptr_update_every + 1)
                elif ema < self.ptr_update_target_flip * 0.5:
                    self.ptr_update_every = max(self.ptr_update_min, self.ptr_update_every - 1)

            # satiety check (optionally soft slice readout)
            if self.soft_readout:
                k = self.soft_readout_k
                offsets = torch.arange(-k, k + 1, device=device, dtype=ptr_float_phys.dtype)
                pos_idx, w, _ = self._compute_kernel_weights(
                    ptr_float_phys, offsets, ring_range, tau_override=self.soft_readout_tau
                )
                pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                gathered = state.gather(1, pos_idx_exp)
                fused = (w.unsqueeze(-1) * gathered.to(w.dtype)).sum(dim=1)
                if fused.dtype != state.dtype:
                    fused = fused.to(state.dtype)
            else:
                gather_idx = last_ptrs.clamp(0, ring_range - 1)
                gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                gathered = state.gather(1, gather_idx_exp)
                fused = gathered.mean(dim=1)
            logits = self.head(fused)
            nan_guard("logits_step", logits, t)
            probs = torch.softmax(logits, dim=1)
            confident = probs.max(dim=1).values > SATIETY_THRESH
            satiety_exited = satiety_exited | confident

        # final readout
        if self.soft_readout:
            k = self.soft_readout_k
            offsets = torch.arange(-k, k + 1, device=device, dtype=ptr_float_phys.dtype)
            pos_idx, w, _ = self._compute_kernel_weights(
                ptr_float_phys, offsets, ring_range, tau_override=self.soft_readout_tau
            )
            pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
            gathered = state.gather(1, pos_idx_exp)
            fused = (w.unsqueeze(-1) * gathered.to(w.dtype)).sum(dim=1)
            if fused.dtype != state.dtype:
                fused = fused.to(state.dtype)
        else:
            gather_idx = last_ptrs.clamp(0, ring_range - 1)
            gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
            gathered = state.gather(1, gather_idx_exp)
            fused = gathered.mean(dim=1)
        logits = self.head(fused)
        nan_guard("logits_final", logits, T)

        self.pointer_hist = hist.detach().cpu()
        self.satiety_exits = int(satiety_exited.sum().item())
        # Expose last pointer bins for MI/TEI evaluation.
        last_bins = torch.bucketize(ptr_int.float(), self.bin_edges.to(device)) - 1
        last_bins = last_bins.clamp(0, self.pointer_hist_bins - 1).detach().cpu()
        self.last_ptr_bins = last_bins
        self.last_ptr_int = ptr_int.detach().cpu()
        # Pointer dynamics summary
        denom = max(1, total_active_steps)
        self.ptr_flip_rate = float(flip_count.sum().item()) / denom
        self.ptr_pingpong_rate = float(pingpong_count.sum().item()) / denom
        self.ptr_max_dwell = int(max_dwell.max().item()) if max_dwell.numel() else 0
        # Mean dwell as active_steps / (flip_count + 1) per sample, averaged.
        mean_dwell = active_steps_per_sample.float() / (flip_count.float() + 1.0)
        self.ptr_mean_dwell = float(mean_dwell.mean().item()) if mean_dwell.numel() else 0.0
        if self.state_loop_metrics:
            mode_denom = max(1, mode_steps)
            self.state_loop_flip_rate = float(mode_flip.sum().item()) / mode_denom
            self.state_loop_abab_rate = float(mode_abab.sum().item()) / mode_denom
            self.state_loop_max_dwell = int(mode_max_dwell.max().item()) if mode_max_dwell.numel() else 0
            self.state_loop_mean_dwell = float(mode_dwell.float().mean().item()) if mode_dwell.numel() else 0.0
            if mode_counts.sum() > 0:
                probs = mode_counts.float() / mode_counts.sum()
                ent = -(probs * torch.log(probs + 1e-12)).sum() / math.log(2.0)
                self.state_loop_entropy = float(ent.item())
            else:
                self.state_loop_entropy = None
        steps_used = max(1, t + 1)
        move_penalty = movement_cost / steps_used
        self.ptr_delta_abs_mean = float(move_penalty)
        return logits, move_penalty


class FileAudioDataset(torch.utils.data.Dataset):
    def __init__(self, items, num_classes, sample_rate=16000, max_len=16000, n_mels=64, max_frames=100):
        self.items = items
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.max_frames = max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.size(1) < self.max_len:
            pad = self.max_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, : self.max_len]
        with torch.no_grad():
            mel = self.melspec(wav)  # [1, n_mels, frames]
            mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0).transpose(0, 1)  # [frames, n_mels]
        if mel.size(0) < self.max_frames:
            pad = self.max_frames - mel.size(0)
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
        else:
            mel = mel[: self.max_frames]
        return mel, label


def _download_zip(url, dest_dir, tag):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{tag}.zip")
    if not os.path.exists(zip_path):
        log(f"Downloading {tag}...")
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def _extract_zip(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def get_fsdd_loader():
    if not HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio not available")
    root = os.path.join(DATA_DIR, "fsdd")
    if OFFLINE_ONLY and not os.path.exists(root):
        raise RuntimeError("offline mode and fsdd not present")
    zip_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
    zip_path = _download_zip(zip_url, root, "fsdd")
    extract_root = os.path.join(root, "free-spoken-digit-dataset-master")
    if not os.path.exists(extract_root):
        _extract_zip(zip_path, root)

    recordings = os.path.join(extract_root, "recordings")
    items = []
    for fname in sorted(os.listdir(recordings)):
        if not fname.endswith(".wav"):
            continue
        label = int(fname.split("_")[0])
        items.append((os.path.join(recordings, fname), label))
    items = items[:MAX_SAMPLES]
    dataset = FileAudioDataset(items, num_classes=10)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return loader, 10


def get_seq_mnist_loader():
    SYNTH_META.clear()
    if os.environ.get("TP6_SYNTH", "0") == "1":
        synth_mode = SYNTH_MODE
        SYNTH_META.update({"enabled": True, "mode": synth_mode, "synth_len": SYNTH_LEN})
        n_samples = max(1, MAX_SAMPLES)
        # Synthetic inputs: [B,256,1]
        x = torch.randint(0, 2, (n_samples, 256, 1), dtype=torch.float32)
        if synth_mode == "markov0":
            y = x[:, -1, 0].to(torch.long)
        elif synth_mode == "markov0_flip":
            y = (1 - x[:, -1, 0]).to(torch.long)
        elif synth_mode == "const0":
            y = torch.zeros((n_samples,), dtype=torch.long)
        elif synth_mode == "hand_kv":
            hand_path = os.environ.get("TP6_HAND_PATH", os.path.join(DATA_DIR, "hand_kv.jsonl"))
            pad_len = int(os.environ.get("TP6_HAND_PAD_LEN", "0"))
            rows = []
            with open(hand_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            if MAX_SAMPLES and MAX_SAMPLES < len(rows):
                rows = rows[:MAX_SAMPLES]
            if len(rows) < HAND_MIN:
                raise RuntimeError(
                    f"hand_kv dataset too small: {len(rows)} rows < HAND_MIN={HAND_MIN} ({hand_path})"
                )
            SYNTH_META.update({"hand_path": hand_path, "rows": len(rows), "pad_len": pad_len})
            log(f"[synth] mode=hand_kv rows={len(rows)} pad_len={pad_len} path={hand_path}")

            xs = []
            ys = []
            for row in rows:
                seq = row.get("x", [])
                label = row.get("y", 0)
                if pad_len > 0:
                    if len(seq) < pad_len:
                        seq = seq + [0] * (pad_len - len(seq))
                    else:
                        seq = seq[:pad_len]
                xs.append(torch.tensor(seq, dtype=torch.float32).view(-1, 1))
                ys.append(int(label))

            class _ListSynth(torch.utils.data.Dataset):
                def __len__(self):
                    return len(xs)

                def __getitem__(self, idx):
                    return xs[idx], ys[idx]

            ds = _ListSynth()

            def collate(batch):
                xs_b, ys_b = zip(*batch)
                return torch.stack(xs_b, dim=0), torch.tensor(ys_b, dtype=torch.long)

            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            num_classes = max(2, max(ys) + 1 if ys else 2)
            return loader, num_classes, collate
        else:
            y = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        SYNTH_META.update({"rows": int(n_samples)})
        log(f"[synth] mode={synth_mode} rows={int(n_samples)}")

        class _Synth(torch.utils.data.Dataset):
            def __len__(self):
                return n_samples

            def __getitem__(self, idx):
                return x[idx], y[idx]

        ds = _Synth()

        def collate(batch):
            xs, ys = zip(*batch)
            return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate,
        )
        return loader, 2, collate

    try:
        import torchvision.transforms as T
        from torchvision.datasets import MNIST
    except Exception as exc:
        raise RuntimeError("torchvision is required for MNIST mode") from exc

    transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
    ds = MNIST(os.path.join(DATA_DIR, "mnist_seq"), train=True, download=not OFFLINE_ONLY, transform=transform)
    if MAX_SAMPLES and MAX_SAMPLES < len(ds):
        ds = Subset(ds, list(range(MAX_SAMPLES)))

    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)  # [B,1,16,16]
        x = x.view(x.size(0), -1, 1)  # [B,256,1]
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )
    return loader, 10, collate


def build_synth_pair_loaders():
    n_samples = max(1, MAX_SAMPLES)
    seq_len = max(1, SYNTH_LEN)
    g = torch.Generator()
    g.manual_seed(SEED)
    x = torch.randint(0, 2, (n_samples, seq_len, 1), dtype=torch.float32, generator=g)
    y_a = x[:, -1, 0].to(torch.long)
    y_b = (1 - x[:, -1, 0]).to(torch.long)

    class _FixedSynth(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            self.xs = xs
            self.ys = ys

        def __len__(self):
            return self.xs.size(0)

        def __getitem__(self, idx):
            return self.xs[idx], self.ys[idx]

    ds_a = _FixedSynth(x, y_a)
    ds_b = _FixedSynth(x, y_b)

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    loader_a = DataLoader(
        ds_a,
        batch_size=BATCH_SIZE,
        shuffle=SYNTH_SHUFFLE,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )
    loader_b = DataLoader(
        ds_b,
        batch_size=BATCH_SIZE,
        shuffle=SYNTH_SHUFFLE,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )
    return loader_a, loader_b, collate


def build_eval_loader_from_subset(train_ds, input_collate=None):
    eval_size = min(EVAL_SAMPLES, len(train_ds))
    if isinstance(train_ds, Subset):
        indices = train_ds.indices[:eval_size]
        eval_subset = Subset(train_ds.dataset, indices)
    else:
        eval_subset = Subset(train_ds, list(range(eval_size)))

    def _collate(batch):
        if input_collate:
            return input_collate(batch)
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    loader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )
    return loader, eval_size


def build_eval_loader_from_dataset(eval_ds, input_collate=None):
    eval_size = min(EVAL_SAMPLES, len(eval_ds))
    eval_subset = Subset(eval_ds, list(range(eval_size)))

    def _collate(batch):
        if input_collate:
            return input_collate(batch)
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    loader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )
    return loader, eval_size


def log_eval_overlap(train_ds, eval_ds, eval_size, label):
    def base_and_indices(ds):
        if isinstance(ds, Subset):
            return ds.dataset, set(ds.indices)
        return ds, None

    train_base, train_idx = base_and_indices(train_ds)
    eval_base, eval_idx = base_and_indices(eval_ds)

    if train_base is eval_base:
        if eval_idx is None:
            overlap = eval_size
        elif train_idx is None:
            overlap = len(eval_idx)
        else:
            overlap = len(train_idx.intersection(eval_idx))
        log(f"[eval] split={label} overlap={overlap}/{eval_size} (shared base dataset)")
    else:
        log(f"[eval] split={label} overlap=0/{eval_size} (disjoint datasets)")


def train_wallclock(model, loader, dataset_name, model_name, num_classes, wall_clock=WALL_CLOCK_SECONDS):
    model = model.to(DEVICE, dtype=DTYPE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = amp_grad_scaler()
    losses = []
    pointer_hist_sum = None
    satiety_exits = 0
    grad_norm = 0.0
    flip_ema = None
    ptr_flip_sum = 0.0
    ptr_mean_dwell_sum = 0.0
    ptr_delta_abs_sum = 0.0
    ptr_max_dwell = 0
    ptr_steps = 0
    panic_reflex = None
    panic_status = ""
    if PANIC_ENABLED:
        panic_reflex = PanicReflex(
            ema_beta=PANIC_BETA,
            panic_threshold=PANIC_THRESHOLD,
            recovery_rate=PANIC_RECOVERY,
            inertia_low=PANIC_INERTIA_LOW,
            inertia_high=PANIC_INERTIA_HIGH,
            walk_prob_max=PANIC_WALK_MAX,
        )

    start = time.time()
    end_time = start + wall_clock if wall_clock > 0 else float("inf")
    last_heartbeat = start
    last_live_trace = start
    step = 0
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError:
            if MOBIUS_ENABLED:
                log("MOBIUS enabled: retrying load_state_dict(strict=False) due to key mismatch")
                missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
                if missing:
                    log(f"MOBIUS load missing keys: {missing}")
                if unexpected:
                    log(f"MOBIUS load unexpected keys: {unexpected}")
            else:
                raise
        if MOBIUS_ENABLED:
            log("MOBIUS enabled: skipping optimizer/scaler load for clean restart")
        else:
            optimizer.load_state_dict(ckpt["optim"])
            if ckpt.get("scaler") and USE_AMP:
                scaler.load_state_dict(ckpt["scaler"])
        step = int(ckpt.get("step", 0))
        losses = list(ckpt.get("losses", []))
        log(f"Resumed from checkpoint: {CHECKPOINT_PATH} (step={step})")
    # cycle loader until wall clock
    stop_early = False
    while time.time() <= end_time:
        for batch in loader:
            if time.time() > end_time:
                break
            inputs, targets = batch
            inputs = inputs.to(DEVICE, non_blocking=True)
            if inputs.dtype != DTYPE:
                inputs = inputs.to(DTYPE)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                outputs, move_pen = model(inputs)
                loss = criterion(outputs, targets) + LAMBDA_MOVE * move_pen
            scaler.scale(loss).backward()
            if GRAD_CLIP > 0.0:
                if USE_AMP:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # Force small kernels to complete and keep the watchdog happy; clear cache frequently.
            if DEVICE == "cuda":
                torch.cuda.synchronize()
                if step % 10 == 0:
                    torch.cuda.empty_cache()

            losses.append(loss.item())
            if LOSS_KEEP > 0 and len(losses) > LOSS_KEEP:
                losses = losses[-LOSS_KEEP:]
            if THERMO_ENABLED and hasattr(model, "ptr_flip_rate") and step % max(1, THERMO_EVERY) == 0:
                flip_ema = apply_thermostat(model, float(model.ptr_flip_rate), flip_ema)
            if panic_reflex is not None:
                ctrl = panic_reflex.update(float(loss))
                model.ptr_inertia = ctrl["inertia"]
                model.ptr_walk_prob = ctrl["walk_prob"]
                panic_status = ctrl["status"]
            if hasattr(model, "pointer_hist"):
                if pointer_hist_sum is None:
                    pointer_hist_sum = model.pointer_hist.clone()
                else:
                    pointer_hist_sum += model.pointer_hist
            if hasattr(model, "satiety_exits"):
                satiety_exits += model.satiety_exits
            if hasattr(model, "ptr_flip_rate"):
                ptr_flip_sum += float(model.ptr_flip_rate)
                ptr_steps += 1
            if hasattr(model, "ptr_mean_dwell"):
                ptr_mean_dwell_sum += float(model.ptr_mean_dwell)
            if hasattr(model, "ptr_delta_abs_mean"):
                ptr_delta_abs_sum += float(model.ptr_delta_abs_mean)
            if hasattr(model, "ptr_max_dwell"):
                ptr_max_dwell = max(ptr_max_dwell, int(model.ptr_max_dwell))
            now = time.time()
            heartbeat_due = (step % HEARTBEAT_STEPS == 0) or (
                HEARTBEAT_SECS > 0.0 and (now - last_heartbeat) >= HEARTBEAT_SECS
            )
            if heartbeat_due and hasattr(model, "theta_ptr_reduced"):
                with torch.no_grad():
                    grad_norm = (
                        model.theta_ptr_reduced.grad.norm().item()
                        if model.theta_ptr_reduced.grad is not None
                        else 0.0
                    )
                log(f"{dataset_name} | {model_name} | grad_norm(theta_ptr)={grad_norm:.4e}")

            if heartbeat_due:
                last_heartbeat = now
                elapsed = now - start
                log(
                    f"{dataset_name} | {model_name} | step {step:04d} | loss {loss.item():.4f} | "
                    f"t={elapsed:.1f}s | ctrl(inertia={model.ptr_inertia:.2f}, deadzone={model.ptr_deadzone:.2f}, walk={model.ptr_walk_prob:.2f})"
                    + (f" | panic={panic_status}" if panic_reflex is not None else "")
                )
                live_due = LIVE_TRACE_EVERY > 0 and (step % LIVE_TRACE_EVERY == 0)
                if HEARTBEAT_SECS > 0.0 and (now - last_live_trace) >= HEARTBEAT_SECS:
                    live_due = True
                if LIVE_TRACE_PATH and len(LIVE_TRACE_PATH) > 0 and live_due:
                    trace = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "step": step,
                        "time_sec": round(elapsed, 3),
                        "loss": loss.item(),
                        "grad_norm_theta_ptr": grad_norm,
                    }
                    if hasattr(model, "ptr_flip_rate"):
                        trace["ptr_flip_rate"] = model.ptr_flip_rate
                        trace["ptr_pingpong_rate"] = model.ptr_pingpong_rate
                        trace["ptr_max_dwell"] = model.ptr_max_dwell
                        trace["ptr_mean_dwell"] = model.ptr_mean_dwell
                        trace["ptr_delta_abs_mean"] = getattr(model, "ptr_delta_abs_mean", None)
                        trace["ptr_inertia"] = model.ptr_inertia
                        trace["ptr_deadzone"] = model.ptr_deadzone
                        trace["ptr_walk_prob"] = model.ptr_walk_prob
                        if panic_reflex is not None:
                            trace["panic_status"] = panic_status
                    if getattr(model, "debug_stats", None):
                        trace.update(model.debug_stats)
                    if hasattr(model, "state_loop_entropy"):
                        trace["state_loop_entropy"] = model.state_loop_entropy
                        trace["state_loop_flip_rate"] = getattr(model, "state_loop_flip_rate", None)
                        trace["state_loop_abab_rate"] = getattr(model, "state_loop_abab_rate", None)
                        trace["state_loop_mean_dwell"] = getattr(model, "state_loop_mean_dwell", None)
                        trace["state_loop_max_dwell"] = getattr(model, "state_loop_max_dwell", None)
                    if DEVICE == "cuda":
                        trace["cuda_mem_alloc_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
                        trace["cuda_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 2)
                    if pointer_hist_sum is not None:
                        hist_np = pointer_hist_sum.cpu().numpy()
                        total = hist_np.sum()
                        if total > 0:
                            probs = hist_np / total
                            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
                        else:
                            entropy = None
                        trace["pointer_entropy"] = entropy
                        trace["pointer_total"] = int(total)
                    try:
                        with open(LIVE_TRACE_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(trace) + "\n")
                    except Exception as e:
                        log(f"live_trace write failed: {e}")
                    last_live_trace = now

            # Explicitly release intermediates to reduce fragmentation risk.
            del outputs, loss
            step += 1
            if MAX_STEPS > 0 and step >= MAX_STEPS:
                stop_early = True
                break
            if SAVE_EVERY_STEPS > 0 and step % SAVE_EVERY_STEPS == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if USE_AMP else None,
                        "step": step,
                        "losses": losses,
                    },
                    CHECKPOINT_PATH,
                )
                log(f"Checkpoint saved @ step {step} -> {CHECKPOINT_PATH}")
        if stop_early:
            break
    # end while

    slope = compute_slope(losses)
    log(f"{dataset_name} | {model_name} | slope {slope:.6f} over {len(losses)} steps")
    ptr_flip_rate = (ptr_flip_sum / ptr_steps) if ptr_steps else None
    ptr_mean_dwell = (ptr_mean_dwell_sum / ptr_steps) if ptr_steps else None
    ptr_delta_abs_mean = (ptr_delta_abs_sum / ptr_steps) if ptr_steps else None
    return {
        "loss_slope": slope,
        "steps": step,
        "losses": losses,
        "pointer_hist": pointer_hist_sum.tolist() if pointer_hist_sum is not None else None,
        "satiety_exits": satiety_exits,
        "ptr_flip_rate": ptr_flip_rate,
        "ptr_pingpong_rate": getattr(model, "ptr_pingpong_rate", None),
        "ptr_max_dwell": ptr_max_dwell,
        "ptr_mean_dwell": ptr_mean_dwell,
        "ptr_delta_abs_mean": ptr_delta_abs_mean,
        "state_loop_entropy": getattr(model, "state_loop_entropy", None),
        "state_loop_flip_rate": getattr(model, "state_loop_flip_rate", None),
        "state_loop_abab_rate": getattr(model, "state_loop_abab_rate", None),
        "state_loop_max_dwell": getattr(model, "state_loop_max_dwell", None),
        "state_loop_mean_dwell": getattr(model, "state_loop_mean_dwell", None),
    }


def eval_model(model, loader, dataset_name, model_name):
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    mi_bins = getattr(model, "pointer_hist_bins", 128)
    joint = torch.zeros((model.head.out_features, mi_bins), dtype=torch.long)
    joint_shuffle = torch.zeros_like(joint) if MI_SHUFFLE else None
    ptr_flip_sum = 0.0
    ptr_steps = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            if inputs.dtype != DTYPE:
                inputs = inputs.to(DTYPE)
            targets = targets.to(DEVICE, non_blocking=True)
            with amp_autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_seen += inputs.size(0)
            if hasattr(model, "last_ptr_bins"):
                bins = model.last_ptr_bins.to(torch.long)
                labels = targets.detach().cpu().to(torch.long)
                idx = labels * mi_bins + bins
                joint += torch.bincount(idx, minlength=joint.numel()).view_as(joint)
                if MI_SHUFFLE:
                    perm = torch.randperm(labels.numel())
                    labels_shuf = labels[perm]
                    idx_shuf = labels_shuf * mi_bins + bins
                    joint_shuffle += torch.bincount(idx_shuf, minlength=joint.numel()).view_as(joint)
            if hasattr(model, "ptr_flip_rate"):
                ptr_flip_sum += float(model.ptr_flip_rate)
                ptr_steps += 1
    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    mi_bits = None
    mi_bits_shuffle = None
    if joint.sum() > 0:
        p = joint.float() / joint.sum()
        pc = p.sum(dim=1, keepdim=True)
        pb = p.sum(dim=0, keepdim=True)
        mi = (p * (torch.log(p + 1e-12) - torch.log(pc * pb + 1e-12))).sum()
        mi_bits = float(mi / math.log(2.0))
    if joint_shuffle is not None and joint_shuffle.sum() > 0:
        p = joint_shuffle.float() / joint_shuffle.sum()
        pc = p.sum(dim=1, keepdim=True)
        pb = p.sum(dim=0, keepdim=True)
        mi = (p * (torch.log(p + 1e-12) - torch.log(pc * pb + 1e-12))).sum()
        mi_bits_shuffle = float(mi / math.log(2.0))
    ptr_flip_rate = (ptr_flip_sum / ptr_steps) if ptr_steps else None
    tei = None
    if mi_bits is not None and ptr_flip_rate is not None:
        tei = acc * mi_bits * (1.0 - ptr_flip_rate)
    log(f"{dataset_name} | {model_name} | eval_loss {avg_loss:.4f} | eval_acc {acc:.4f} | eval_n {total_seen}")
    return {
        "eval_loss": avg_loss,
        "eval_acc": acc,
        "eval_n": total_seen,
        "eval_mi_bits": mi_bits,
        "eval_mi_bits_shuffled": mi_bits_shuffle,
        "eval_ptr_flip_rate": ptr_flip_rate,
        "eval_tei": tei,
    }


def train_steps(model, loader, steps, dataset_name, model_name):
    model = model.to(DEVICE, dtype=DTYPE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = amp_grad_scaler()
    it = iter(loader)
    step = 0
    pointer_hist_sum = None
    satiety_exits = 0
    losses = []
    ptr_flip_sum = 0.0
    ptr_mean_dwell_sum = 0.0
    ptr_delta_abs_sum = 0.0
    ptr_max_dwell = 0
    ptr_steps = 0
    flip_ema = None
    panic_reflex = None
    panic_status = ""
    if PANIC_ENABLED:
        panic_reflex = PanicReflex(
            ema_beta=PANIC_BETA,
            panic_threshold=PANIC_THRESHOLD,
            recovery_rate=PANIC_RECOVERY,
            inertia_low=PANIC_INERTIA_LOW,
            inertia_high=PANIC_INERTIA_HIGH,
            walk_prob_max=PANIC_WALK_MAX,
        )
    while step < steps:
        try:
            inputs, targets = next(it)
        except StopIteration:
            it = iter(loader)
            inputs, targets = next(it)
        inputs = inputs.to(DEVICE, non_blocking=True)
        if inputs.dtype != DTYPE:
            inputs = inputs.to(DTYPE)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast():
            outputs, move_pen = model(inputs)
            loss = criterion(outputs, targets) + LAMBDA_MOVE * move_pen
        scaler.scale(loss).backward()
        if GRAD_CLIP > 0.0:
            if USE_AMP:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
            if step % 10 == 0:
                torch.cuda.empty_cache()

        losses.append(loss.item())
        if hasattr(model, "pointer_hist"):
            if pointer_hist_sum is None:
                pointer_hist_sum = model.pointer_hist.clone()
            else:
                pointer_hist_sum += model.pointer_hist
        if hasattr(model, "satiety_exits"):
            satiety_exits += model.satiety_exits
        if hasattr(model, "ptr_flip_rate"):
            ptr_flip_sum += float(model.ptr_flip_rate)
            ptr_steps += 1
            if THERMO_ENABLED and step % max(1, THERMO_EVERY) == 0:
                flip_ema = apply_thermostat(model, float(model.ptr_flip_rate), flip_ema)
        if panic_reflex is not None:
            ctrl = panic_reflex.update(float(loss))
            model.ptr_inertia = ctrl["inertia"]
            model.ptr_walk_prob = ctrl["walk_prob"]
            panic_status = ctrl["status"]
        if hasattr(model, "ptr_mean_dwell"):
            ptr_mean_dwell_sum += float(model.ptr_mean_dwell)
        if hasattr(model, "ptr_delta_abs_mean"):
            ptr_delta_abs_sum += float(model.ptr_delta_abs_mean)
        if hasattr(model, "ptr_max_dwell"):
            ptr_max_dwell = max(ptr_max_dwell, int(model.ptr_max_dwell))
        del outputs, loss
        step += 1
    slope = compute_slope(losses)
    ptr_flip_rate = (ptr_flip_sum / ptr_steps) if ptr_steps else None
    ptr_mean_dwell = (ptr_mean_dwell_sum / ptr_steps) if ptr_steps else None
    ptr_delta_abs_mean = (ptr_delta_abs_sum / ptr_steps) if ptr_steps else None
    return {
        "loss_slope": slope,
        "steps": steps,
        "pointer_hist": pointer_hist_sum.tolist() if pointer_hist_sum is not None else None,
        "satiety_exits": satiety_exits,
        "ptr_flip_rate": ptr_flip_rate,
        "ptr_mean_dwell": ptr_mean_dwell,
        "ptr_max_dwell": ptr_max_dwell,
        "ptr_delta_abs_mean": ptr_delta_abs_mean,
    }


def mutate_state_dict(parent_state, std=EVO_MUT_STD):
    child = {}
    for k, v in parent_state.items():
        if not torch.is_floating_point(v):
            child[k] = v.clone()
            continue
        noise = torch.randn_like(v, device="cpu") * std
        child[k] = (v.cpu() + noise).to(v.dtype)
    return child


def run_evolution(dataset_name, loader, eval_loader, input_dim, num_classes):
    log(f"=== Evolution mode | dataset={dataset_name} | pop={EVO_POP} gens={EVO_GENS} steps/ind={EVO_STEPS} ===")
    # init population
    population = []
    for i in range(EVO_POP):
        m = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=8)
        population.append(m)

    best_eval = None
    for gen in range(EVO_GENS):
        gen_fitness = []
        for idx, model in enumerate(population):
            train_stats = train_steps(model, loader, EVO_STEPS, dataset_name, f"evo_{gen}_{idx}")
            eval_stats = eval_model(model, eval_loader, dataset_name, f"evo_{gen}_{idx}")
            fitness = 1.0 - eval_stats["eval_loss"]
            gen_fitness.append((fitness, model, train_stats, eval_stats))

        gen_fitness.sort(key=lambda x: x[0], reverse=True)
        topk = max(1, EVO_POP // 3)
        elites = gen_fitness[:topk]
        if best_eval is None or elites[0][0] > best_eval[0]:
            best_eval = elites[0]
        log(f"Gen {gen}: best_acc={elites[0][3]['eval_acc']:.4f}, loss={elites[0][3]['eval_loss']:.4f}")

        # Refill population
        new_population = [e[1] for e in elites]  # keep elites
        while len(new_population) < EVO_POP:
            parent = random.choice(elites)[1]
            child = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=8)
            child.load_state_dict(mutate_state_dict(parent.state_dict(), std=EVO_MUT_STD))
            new_population.append(child)
        population = new_population

    # final best eval stats already stored in best_eval
    fitness, best_model, train_stats, eval_stats = best_eval
    return {
        "mode": "evolution",
        "best_train": train_stats,
        "best_eval": eval_stats,
        "best_fitness": fitness,
    }


def run_phase(dataset_name: str, loader, eval_loader, input_dim: int, num_classes: int):
    extra = ""
    if SYNTH_META.get("enabled"):
        extra = f" | synth_mode={SYNTH_META.get('mode')}"
    log(f"=== Phase 6.5 | dataset={dataset_name} | num_classes={num_classes}{extra} ===")
    hallway = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=8)
    hall_train = train_wallclock(hallway, loader, dataset_name, "absolute_hallway", num_classes)
    hall_eval = eval_model(hallway, eval_loader, dataset_name, "absolute_hallway")
    result = {"dataset": dataset_name, "absolute_hallway": {"train": hall_train, "eval": hall_eval}}
    if SYNTH_META:
        result["meta"] = dict(SYNTH_META)
    return result


def run_lockout_test():
    log("=== Lockout test | deterministic synth A->B (label flip) ===")
    loader_a, loader_b, collate = build_synth_pair_loaders()
    eval_a, _ = build_eval_loader_from_subset(loader_a.dataset, input_collate=collate)
    eval_b, _ = build_eval_loader_from_subset(loader_b.dataset, input_collate=collate)

    model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=RING_LEN, slot_dim=8)

    train_a = train_steps(model, loader_a, PHASE_A_STEPS, "synthA", "absolute_hallway")
    eval_a_post = eval_model(model, eval_a, "synthA", "absolute_hallway")
    eval_b_pre = eval_model(model, eval_b, "synthB_pre", "absolute_hallway")

    train_b = train_steps(model, loader_b, PHASE_B_STEPS, "synthB", "absolute_hallway")
    eval_b_post = eval_model(model, eval_b, "synthB_post", "absolute_hallway")
    eval_a_post_b = eval_model(model, eval_a, "synthA_postB", "absolute_hallway")

    return {
        "mode": "lockout",
        "phase_a": {"train": train_a, "eval": eval_a_post},
        "phase_b": {"pre_eval": eval_b_pre, "train": train_b, "eval": eval_b_post},
        "forgetting_check": eval_a_post_b,
        "meta": {
            "phase_a_steps": PHASE_A_STEPS,
            "phase_b_steps": PHASE_B_STEPS,
            "synth_len": SYNTH_LEN,
            "synth_shuffle": SYNTH_SHUFFLE,
        },
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not RESUME:
        rotate_artifacts()
    set_seed(SEED)
    # Reduce kernel search overhead / variance.
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    log(f"Phase 6.5 Absolute Hallway start | device={DEVICE} | offline_only={OFFLINE_ONLY}")

    summary = []

    # Sequential MNIST (pixel-by-pixel 16x16 = 256 steps)
    if RUN_MODE == "lockout":
        summary.append(run_lockout_test())
    else:
        mnist_loader, mnist_classes, mnist_collate = get_seq_mnist_loader()
        eval_label = "train_subset"
        if SYNTH_META.get("enabled") or EVAL_SPLIT == "subset":
            mnist_eval_loader, eval_size = build_eval_loader_from_subset(
                mnist_loader.dataset, input_collate=mnist_collate
            )
            eval_label = "train_subset"
        else:
            try:
                import torchvision.transforms as T
                from torchvision.datasets import MNIST
                transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
                eval_ds = MNIST(
                    os.path.join(DATA_DIR, "mnist_seq"),
                    train=False,
                    download=not OFFLINE_ONLY,
                    transform=transform,
                )
                mnist_eval_loader, eval_size = build_eval_loader_from_dataset(
                    eval_ds, input_collate=mnist_collate
                )
                eval_label = "mnist_test"
            except Exception as exc:
                log(f"[eval] test split unavailable ({exc}); falling back to train subset")
                mnist_eval_loader, eval_size = build_eval_loader_from_subset(
                    mnist_loader.dataset, input_collate=mnist_collate
                )
                eval_label = "train_subset_fallback"
        log_eval_overlap(mnist_loader.dataset, mnist_eval_loader.dataset, eval_size, eval_label)
        if RUN_MODE == "evolution":
            summary.append(
                run_evolution("seq_mnist", mnist_loader, mnist_eval_loader, input_dim=1, num_classes=mnist_classes)
            )
        else:
            summary.append(run_phase("seq_mnist", mnist_loader, mnist_eval_loader, input_dim=1, num_classes=mnist_classes))

    # Ensure GPU work is complete before writing summary to avoid partial/hung writes.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    summary_path = os.path.join(ROOT, "summaries", "current", "tournament_phase6_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Tournament done. Summary saved to {summary_path}")
    sync_current_to_last()


if __name__ == "__main__":
    main()
