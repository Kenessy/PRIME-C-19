#!/usr/bin/env python
import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tournament_phase6 as tp6
from tournament_phase6 import AbsoluteHallway


@dataclass
class BenchResult:
    dataset: str
    activation: str
    task: str
    eval_loss: float
    eval_acc: float | None
    eval_mse: float | None


class ArrayDataset(Dataset):
    def __init__(self, x, y, task):
        self.x = x
        self.y = y
        self.task = task

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.task == "regression":
            return self.x[idx], self.y[idx]
        return self.x[idx], int(self.y[idx])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def next_gaussian(rng):
    u1 = 1.0 - rng.random()
    u2 = 1.0 - rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def generate_xor(rng, samples, noise):
    x = rng.uniform(-1.0, 1.0, size=(samples, 2))
    x += rng.normal(0.0, noise, size=x.shape)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int64)
    return x, y


def generate_two_moons(rng, samples, noise):
    half = samples // 2
    t = rng.uniform(0.0, math.pi, size=half)
    x1 = np.cos(t)
    y1 = np.sin(t)
    x1 += rng.normal(0.0, noise, size=x1.shape)
    y1 += rng.normal(0.0, noise, size=y1.shape)
    x2 = 1.0 - np.cos(t)
    y2 = -np.sin(t) - 0.5
    x2 += rng.normal(0.0, noise, size=x2.shape)
    y2 += rng.normal(0.0, noise, size=y2.shape)
    x = np.stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])], axis=1)
    y = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(samples - half, dtype=np.int64)])
    return x, y


def generate_circles(rng, samples, noise):
    half = samples // 2
    r1 = 0.5 + rng.uniform(0.0, 0.1, size=half)
    t1 = rng.uniform(0.0, math.pi * 2.0, size=half)
    x1 = r1 * np.cos(t1)
    y1 = r1 * np.sin(t1)
    x1 += rng.normal(0.0, noise, size=x1.shape)
    y1 += rng.normal(0.0, noise, size=y1.shape)
    r2 = 1.0 + rng.uniform(0.0, 0.2, size=samples - half)
    t2 = rng.uniform(0.0, math.pi * 2.0, size=samples - half)
    x2 = r2 * np.cos(t2)
    y2 = r2 * np.sin(t2)
    x2 += rng.normal(0.0, noise, size=x2.shape)
    y2 += rng.normal(0.0, noise, size=y2.shape)
    x = np.stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])], axis=1)
    y = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(samples - half, dtype=np.int64)])
    return x, y


def generate_spiral(rng, samples, classes, noise):
    inputs = np.zeros((samples, 2), dtype=np.float64)
    labels = np.zeros((samples,), dtype=np.int64)
    per_class = samples // classes
    index = 0
    for c in range(classes):
        for i in range(per_class):
            r = i / per_class
            t = (c * 4.0) + (r * 4.0) + (rng.random() * noise)
            inputs[index] = [r * math.cos(t), r * math.sin(t)]
            labels[index] = c
            index += 1
    while index < samples:
        inputs[index] = rng.uniform(-1.0, 1.0, size=2)
        labels[index] = 0
        index += 1
    return inputs, labels


def generate_sine(rng, samples, noise):
    x = (rng.uniform(-1.0, 1.0, size=(samples, 1)) * math.pi).astype(np.float64)
    y = np.sin(x) + rng.normal(0.0, noise, size=x.shape)
    return x, y


def to_sequence(x, mode):
    if mode == "steps2":
        seq = np.stack([x[:, 0], x[:, 1]], axis=1)
        return torch.tensor(seq[:, :, None], dtype=torch.float32)
    return torch.tensor(x[:, None, :], dtype=torch.float32)


def split_data(rng, x, y, train_frac):
    idx = rng.permutation(x.shape[0])
    split = int(train_frac * x.shape[0])
    train_idx = idx[:split]
    eval_idx = idx[split:]
    return x[train_idx], y[train_idx], x[eval_idx], y[eval_idx]


def build_model(input_dim, num_classes, activation, ring_len, slot_dim, gauss_k, gauss_tau, device):
    model = AbsoluteHallway(
        input_dim=input_dim,
        num_classes=num_classes,
        ring_len=ring_len,
        slot_dim=slot_dim,
        ptr_stride=1,
        gauss_k=gauss_k,
        gauss_tau=gauss_tau,
    ).to(device)
    model.act_name = activation
    model.ptr_inertia = 0.0
    model.ptr_deadzone = 0.0
    model.ptr_walk_prob = 0.10
    model.ptr_update_every = 1
    model.ptr_warmup_steps = 0
    return model


def train_model(model, loader, task, epochs, lr, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if task == "regression":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if task == "regression":
                targets = targets.to(device)
            else:
                targets = targets.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(inputs)
            if task == "regression":
                loss = criterion(outputs.squeeze(1), targets.squeeze(1))
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def eval_model(model, loader, task, device):
    model.eval()
    if task == "regression":
        criterion = torch.nn.MSELoss(reduction="sum")
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs.squeeze(1), targets.squeeze(1))
                total_loss += float(loss.item())
                total += inputs.size(0)
        mse = total_loss / max(1, total)
        return mse, None
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item())
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total += inputs.size(0)
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Small PRIME C-19 synthetic benchmark.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ring-len", type=int, default=128)
    parser.add_argument("--slot-dim", type=int, default=16)
    parser.add_argument("--gauss-k", type=int, default=2)
    parser.add_argument("--gauss-tau", type=float, default=2.0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seq-mode", choices=["steps2", "flat"], default="steps2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--activations", default="c19,relu,silu")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tp6.SATIETY_THRESH = 1.1
    tp6.STATE_DECAY = 0.99
    tp6.STATE_CLIP = 2.0
    tp6.UPDATE_SCALE = 1.0
    tp6.PTR_JUMP_DISABLED = True
    tp6.PTR_JUMP_CAP = 1.0
    tp6.PTR_UPDATE_EVERY = 1
    tp6.PTR_WALK_PROB = 0.10

    device = torch.device(args.device)

    datasets = []
    x, y = generate_xor(rng, 512, noise=0.2)
    datasets.append(("xor", "classification", x, y, 2))
    x, y = generate_two_moons(rng, 768, noise=0.12)
    datasets.append(("two_moons", "classification", x, y, 2))
    x, y = generate_circles(rng, 768, noise=0.1)
    datasets.append(("circles", "classification", x, y, 2))
    x, y = generate_spiral(rng, 900, classes=3, noise=0.18)
    datasets.append(("spiral", "classification", x, y, 3))
    x, y = generate_sine(rng, 512, noise=0.05)
    datasets.append(("sine", "regression", x, y, 1))

    activations = [a.strip() for a in args.activations.split(",") if a.strip()]

    results = []
    for name, task, x, y, num_classes in datasets:
        if task == "classification":
            x_train, y_train, x_eval, y_eval = split_data(rng, x, y, args.train_frac)
            x_train = to_sequence(x_train, args.seq_mode)
            x_eval = to_sequence(x_eval, args.seq_mode)
            if args.seq_mode == "steps2":
                input_dim = 1
            else:
                input_dim = 2
            train_ds = ArrayDataset(x_train, y_train, task)
            eval_ds = ArrayDataset(x_eval, y_eval, task)
        else:
            x_train, y_train, x_eval, y_eval = split_data(rng, x, y, args.train_frac)
            x_train = torch.tensor(x_train[:, None, :], dtype=torch.float32)
            x_eval = torch.tensor(x_eval[:, None, :], dtype=torch.float32)
            input_dim = 1
            train_ds = ArrayDataset(x_train, torch.tensor(y_train, dtype=torch.float32), task)
            eval_ds = ArrayDataset(x_eval, torch.tensor(y_eval, dtype=torch.float32), task)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

        for act in activations:
            model = build_model(
                input_dim=input_dim,
                num_classes=num_classes,
                activation=act,
                ring_len=args.ring_len,
                slot_dim=args.slot_dim,
                gauss_k=args.gauss_k,
                gauss_tau=args.gauss_tau,
                device=device,
            )
            train_model(model, train_loader, task, args.epochs, args.lr, device)
            eval_loss, eval_acc = eval_model(model, eval_loader, task, device)
            if task == "regression":
                results.append(
                    BenchResult(name, act, task, eval_loss, None, eval_loss)
                )
                print(
                    f"{name:<10} | {act:<6} | mse={eval_loss:.4f}"
                )
            else:
                results.append(
                    BenchResult(name, act, task, eval_loss, eval_acc, None)
                )
                print(
                    f"{name:<10} | {act:<6} | loss={eval_loss:.4f} acc={eval_acc:.4f}"
                )


if __name__ == "__main__":
    main()
