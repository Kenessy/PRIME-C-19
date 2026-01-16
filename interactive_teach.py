import argparse
import importlib
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def build_vocab() -> Dict[str, int]:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\"-()?"
    vocab: Dict[str, int] = {}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, seq_len: int, vocab: Dict[str, int]) -> torch.Tensor:
    text = text.lower()
    vocab_size = len(vocab)
    x = torch.zeros(seq_len, vocab_size, dtype=torch.float32)
    for i, ch in enumerate(text[:seq_len]):
        idx = vocab.get(ch, vocab.get("?", 0))
        x[i, idx] = 1.0
    return x


def encode_numeric(text: str, seq_len: int) -> torch.Tensor:
    parts = text.replace(",", " ").split()
    values: List[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            continue
    x = torch.zeros(seq_len, 1, dtype=torch.float32)
    for i, val in enumerate(values[:seq_len]):
        x[i, 0] = val
    return x


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive teach loop for AbsoluteHallway.")
    parser.add_argument("--mode", choices=["text", "numeric"], default="text")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--act", default="identity")
    parser.add_argument("--gauss-k", type=int, default=2)
    parser.add_argument("--gauss-tau", type=float, default=0.5)
    parser.add_argument("--ring-len", type=int, default=2048)
    parser.add_argument("--slot-dim", type=int, default=8)
    parser.add_argument("--ptr-inertia", type=float, default=0.0)
    parser.add_argument("--ptr-deadzone", type=float, default=0.0)
    parser.add_argument("--ptr-walk-prob", type=float, default=0.2)
    parser.add_argument("--ptr-no-round", action="store_true")
    parser.add_argument("--lmove", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--replay", type=int, default=16)
    parser.add_argument("--replay-max", type=int, default=512)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save", default="")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--auto-steps", type=int, default=200)
    parser.add_argument("--auto-batch", type=int, default=8)
    parser.add_argument("--auto-print-every", type=int, default=1)
    parser.add_argument("--teacher", choices=["english_noise", "numeric_sum"], default="english_noise")
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--label-flip", action="store_true")
    parser.add_argument("--panic-reflex", action="store_true")
    parser.add_argument("--panic-threshold", type=float, default=1.5)
    parser.add_argument("--panic-beta", type=float, default=0.9)
    parser.add_argument("--panic-recovery", type=float, default=0.01)
    parser.add_argument("--panic-inertia-low", type=float, default=0.1)
    parser.add_argument("--panic-inertia-high", type=float, default=0.95)
    parser.add_argument("--panic-walk-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def apply_env(args: argparse.Namespace, device: str) -> None:
    os.environ["TP6_DEVICE"] = device
    os.environ["TP6_PRECISION"] = args.precision
    os.environ["TP6_ACT"] = args.act
    os.environ["TP6_GAUSS_K"] = str(args.gauss_k)
    os.environ["TP6_GAUSS_TAU"] = str(args.gauss_tau)
    os.environ["TP6_RING_LEN"] = str(args.ring_len)
    os.environ["TP6_PTR_INERTIA"] = str(args.ptr_inertia)
    os.environ["TP6_PTR_DEADZONE"] = str(args.ptr_deadzone)
    os.environ["TP6_PTR_WALK_PROB"] = str(args.ptr_walk_prob)
    os.environ["TP6_PTR_NO_ROUND"] = "1" if args.ptr_no_round else "0"
    os.environ["TP6_LMOVE"] = str(args.lmove)
    os.environ["TP6_GRAD_CLIP"] = str(args.grad_clip)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    if not path:
        return 0
    if not os.path.exists(path):
        print(f"[warn] checkpoint not found: {path}")
        return 0
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data.get("model", data))
    if "optim" in data:
        optimizer.load_state_dict(data["optim"])
    return int(data.get("step", 0))


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    if not path:
        return
    payload = {"model": model.state_dict(), "optim": optimizer.state_dict(), "step": step}
    torch.save(payload, path)


def make_english_sentence(rng: random.Random) -> str:
    words = [
        "the", "a", "an", "small", "big", "blue", "green", "quick", "slow",
        "cat", "dog", "bird", "car", "street", "rain", "light", "house",
        "walks", "runs", "jumps", "stops", "moves", "looks", "finds",
        "on", "in", "under", "near", "over", "with", "without",
    ]
    length = rng.randint(4, 9)
    sent = " ".join(rng.choice(words) for _ in range(length))
    if rng.random() < 0.3:
        sent += rng.choice([".", "!", "?"])
    return sent


def make_noise_sentence(rng: random.Random, length: int = 32) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_numeric_sequence(rng: random.Random, seq_len: int) -> Tuple[str, int]:
    values = [rng.randint(-9, 9) for _ in range(seq_len)]
    label = 1 if sum(values) >= 0 else 0
    text = " ".join(str(v) for v in values)
    return text, label


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device.strip().lower()
    if device not in {"cuda", "cpu"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    apply_env(args, device)

    tp6 = importlib.import_module("tournament_phase6")
    torch.set_default_dtype(tp6.DTYPE)

    vocab: Dict[str, int] = {}
    if args.mode == "text":
        vocab = build_vocab()
        input_dim = len(vocab)
    else:
        input_dim = 1

    model = tp6.AbsoluteHallway(
        input_dim=input_dim,
        num_classes=args.classes,
        ring_len=args.ring_len,
        slot_dim=args.slot_dim,
        ptr_stride=1,
        gauss_k=args.gauss_k,
        gauss_tau=args.gauss_tau,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    step = load_checkpoint(args.checkpoint, model, optimizer)

    replay: List[Tuple[torch.Tensor, int]] = []
    flip_ema = None
    panic_reflex = None
    panic_status = ""
    if args.panic_reflex:
        panic_reflex = PanicReflex(
            ema_beta=args.panic_beta,
            panic_threshold=args.panic_threshold,
            recovery_rate=args.panic_recovery,
            inertia_low=args.panic_inertia_low,
            inertia_high=args.panic_inertia_high,
            walk_prob_max=args.panic_walk_prob,
        )
    if args.auto:
        rng = random.Random(args.seed)
        for auto_step in range(args.auto_steps):
            batch: List[Tuple[torch.Tensor, int]] = []
            samples: List[Tuple[str, int]] = []
            for _ in range(args.auto_batch):
                if args.mode == "text":
                    if args.teacher == "english_noise":
                        if rng.random() < args.noise_prob:
                            text = make_noise_sentence(rng, length=args.seq_len)
                            label = 0
                        else:
                            text = make_english_sentence(rng)
                            label = 1
                    else:
                        text = make_english_sentence(rng)
                        label = 1
                    x = encode_text(text, args.seq_len, vocab)
                else:
                    text, label = make_numeric_sequence(rng, args.seq_len)
                    x = encode_numeric(text, args.seq_len)
                if args.label_flip and args.classes > 1:
                    label = (args.classes - 1) - label
                samples.append((text, label))
                batch.append((x, label))
            for _ in range(min(args.replay, len(replay))):
                batch.append(random.choice(replay))
            replay.extend([(b[0].clone(), b[1]) for b in batch[: args.auto_batch]])
            if len(replay) > args.replay_max:
                replay = replay[-args.replay_max :]

            xb = torch.stack([item[0] for item in batch]).to(device)
            yb = torch.tensor([item[1] for item in batch], device=device)

            model.train()
            start = time.time()
            logits, move_pen = model(xb)
            loss = criterion(logits, yb)
            if args.lmove > 0.0:
                loss = loss + args.lmove * move_pen
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            dt = time.time() - start
            step += 1
            if getattr(tp6, "THERMO_ENABLED", False) and hasattr(model, "ptr_flip_rate"):
                if step % max(1, tp6.THERMO_EVERY) == 0:
                    flip_ema = tp6.apply_thermostat(model, float(model.ptr_flip_rate), flip_ema)
            if panic_reflex is not None:
                ctrl = panic_reflex.update(float(loss))
                model.ptr_inertia = ctrl["inertia"]
                model.ptr_walk_prob = ctrl["walk_prob"]
                panic_status = ctrl["status"]

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = float((preds == yb).float().mean().item())
                probs = torch.softmax(logits[0], dim=0).cpu()
                pred0 = int(torch.argmax(probs).item())
                topk = torch.topk(probs, k=min(3, probs.numel()))
                top_items = ", ".join([f"{int(i)}:{float(v):.3f}" for v, i in zip(topk.values, topk.indices)])

            if (auto_step + 1) % max(1, args.auto_print_every) == 0:
                panic_note = f" panic={panic_status}" if panic_reflex is not None else ""
                print(
                    f"[auto {auto_step+1:04d}] loss={float(loss):.4f} acc={acc:.3f} "
                    f"pred0={pred0} top3=[{top_items}] "
                    f"flip={getattr(model, 'ptr_flip_rate', float('nan')):.3f} "
                    f"dwell={getattr(model, 'ptr_mean_dwell', float('nan')):.2f} "
                    f"ctrl(inertia={model.ptr_inertia:.2f}, deadzone={model.ptr_deadzone:.2f}, walk={model.ptr_walk_prob:.2f}){panic_note} "
                    f"t={dt:.3f}s"
                )
                if args.print_samples and samples:
                    text, label = samples[0]
                    print(f"[sample] label={label} text='{text[:80]}'")

            if args.save and step % 10 == 0:
                save_checkpoint(args.save, model, optimizer, step)
        return

    print("Interactive teach mode. Commands: :quit, :save, :load <path>, :stats")
    while True:
        prompt = "text> " if args.mode == "text" else "nums> "
        raw = input(prompt).strip()
        if not raw:
            continue
        if raw.startswith(":"):
            cmd = raw.split()
            if cmd[0] in {":quit", ":exit"}:
                break
            if cmd[0] == ":save":
                path = args.save or args.checkpoint or "interactive_checkpoint.pt"
                save_checkpoint(path, model, optimizer, step)
                print(f"[save] {path}")
                continue
            if cmd[0] == ":load" and len(cmd) > 1:
                step = load_checkpoint(cmd[1], model, optimizer)
                print(f"[load] {cmd[1]}")
                continue
            if cmd[0] == ":stats":
                print(
                    f"[stats] flip={getattr(model, 'ptr_flip_rate', float('nan')):.3f} "
                    f"dwell_mean={getattr(model, 'ptr_mean_dwell', float('nan')):.2f} "
                    f"dwell_max={getattr(model, 'ptr_max_dwell', 0)}"
                )
                continue
            print("Commands: :quit, :save, :load <path>, :stats")
            continue

        label_raw = input("label> ").strip()
        if label_raw == "":
            label = None
        else:
            try:
                label = int(label_raw)
            except ValueError:
                print("[warn] label must be int")
                continue
        if args.label_flip and label is not None and args.classes > 1:
            label = (args.classes - 1) - label

        if args.mode == "text":
            x = encode_text(raw, args.seq_len, vocab)
        else:
            x = encode_numeric(raw, args.seq_len)

        if label is not None:
            replay.append((x.clone(), label))
            if len(replay) > args.replay_max:
                replay.pop(0)

        batch: List[Tuple[torch.Tensor, int]] = []
        if label is not None:
            batch.append((x, label))
        if args.replay > 0 and replay:
            for _ in range(min(args.replay, len(replay))):
                batch.append(random.choice(replay))

        if not batch:
            batch = [(x, 0)]

        xb = torch.stack([item[0] for item in batch]).to(device)
        yb = torch.tensor([item[1] for item in batch], device=device)

        model.train()
        start = time.time()
        logits, move_pen = model(xb)
        loss = criterion(logits, yb)
        if args.lmove > 0.0:
            loss = loss + args.lmove * move_pen
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        dt = time.time() - start
        step += 1

        with torch.no_grad():
            probs = torch.softmax(logits[0], dim=0).cpu()
            pred = int(torch.argmax(probs).item())
            topk = torch.topk(probs, k=min(3, probs.numel()))
            top_items = ", ".join([f"{int(i)}:{float(v):.3f}" for v, i in zip(topk.values, topk.indices)])

        print(
            f"[step {step:05d}] loss={float(loss):.4f} pred={pred} top3=[{top_items}] "
            f"flip={getattr(model, 'ptr_flip_rate', float('nan')):.3f} "
            f"dwell={getattr(model, 'ptr_mean_dwell', float('nan')):.2f} "
            f"t={dt:.3f}s"
        )

        if args.save and step % 10 == 0:
            save_checkpoint(args.save, model, optimizer, step)


if __name__ == "__main__":
    main()
