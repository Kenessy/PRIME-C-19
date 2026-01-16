import os
import time
import math
import json
import random
import csv
import zipfile
import urllib.request
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T

import torchaudio


ROOT = r"G:\AI\pilot_pulse"
DATA_DIR = os.path.join(ROOT, "data")
LOG_PATH = os.path.join(ROOT, "pilot_pulse.log")

SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFLINE_ONLY = os.environ.get("PILOT_OFFLINE", "1") == "1"

MAX_SAMPLES = 5000
EVAL_SAMPLES = 1024
BATCH_SIZE = 128
MAX_STEPS = 120  # per model
LR = 1e-3
WALL_CLOCK_SECONDS = 15 * 60  # 15 minutes


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


def compute_slope(losses):
    if len(losses) < 2:
        return float("nan")
    x = np.arange(len(losses), dtype=np.float64)
    y = np.array(losses, dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    return float(a)


class VisionDavidGRU(nn.Module):
    def __init__(self, hidden=128, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=96, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: [B, 3, 32, 32] -> sequence of 32 rows, each 96 dims
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, 32, 3, 32]
        x = x.view(x.size(0), 32, 96)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)


class VisionSwarmGRU(nn.Module):
    def __init__(self, hidden=128, num_units=8, num_classes=10):
        super().__init__()
        self.num_units = num_units
        self.units = nn.ModuleList(
            [nn.GRU(input_size=96, hidden_size=hidden, batch_first=True) for _ in range(num_units)]
        )
        self.fc = nn.Linear(hidden * num_units, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), 32, 96)
        outs = []
        for gru in self.units:
            out, _ = gru(x)
            outs.append(out[:, -1, :])
        fused = torch.cat(outs, dim=1)
        return self.fc(fused)


class AudioDavidGRU(nn.Module):
    def __init__(self, hidden=128, num_classes=35, n_mels=64):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.gru = nn.GRU(input_size=n_mels, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, wav):
        # wav: [B, 1, 16000]
        with torch.no_grad():
            mel = self.melspec(wav)  # [B, n_mels, time]
            mel = torch.log(mel + 1e-6)
        mel = mel.transpose(1, 2).contiguous()  # [B, time, n_mels]
        out, _ = self.gru(mel)
        last = out[:, -1, :]
        return self.fc(last)


class AudioSwarmGRU(nn.Module):
    def __init__(self, hidden=128, num_units=8, num_classes=35, n_mels=64):
        super().__init__()
        self.num_units = num_units
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.units = nn.ModuleList(
            [nn.GRU(input_size=n_mels, hidden_size=hidden, batch_first=True) for _ in range(num_units)]
        )
        self.fc = nn.Linear(hidden * num_units, num_classes)

    def forward(self, wav):
        with torch.no_grad():
            mel = self.melspec(wav)
            mel = torch.log(mel + 1e-6)
        mel = mel.transpose(1, 2).contiguous()
        outs = []
        for gru in self.units:
            out, _ = gru(mel)
            outs.append(out[:, -1, :])
        fused = torch.cat(outs, dim=1)
        return self.fc(fused)


class SeqSwarmGRU(nn.Module):
    def __init__(self, input_dim=1, hidden=128, num_units=8, num_classes=10):
        super().__init__()
        self.units = nn.ModuleList(
            [nn.GRU(input_size=input_dim, hidden_size=hidden, batch_first=True) for _ in range(num_units)]
        )
        self.fc = nn.Linear(hidden * num_units, num_classes)

    def forward(self, x):
        # x: [B, T, input_dim]
        outs = []
        for gru in self.units:
            out, _ = gru(x)
            outs.append(out[:, -1, :])
        fused = torch.cat(outs, dim=1)
        return self.fc(fused)


class ChaosClock(nn.Module):
    def __init__(self, input_dim, num_classes, ring_len=4096, slot_dim=8, teleporters=None):
        super().__init__()
        self.ring_len = ring_len
        self.slot_dim = slot_dim
        self.teleporters = teleporters or [0, 1024, 2048, 3072]
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.jump_score = nn.Linear(slot_dim, 1)
        self.head = nn.Linear(slot_dim * len(self.teleporters), num_classes)
        self.register_buffer("teleporter_tensor", torch.tensor(self.teleporters, dtype=torch.long))
        self.jump_hist = torch.zeros(len(self.teleporters), dtype=torch.long)

    def forward(self, x):
        # x: [B, T, input_dim]
        B, T, _ = x.shape
        device = x.device
        state = x.new_zeros(B, self.ring_len, self.slot_dim)
        ptr = torch.zeros(B, dtype=torch.long, device=device)
        jump_hist_local = torch.zeros(len(self.teleporters), device=device)

        tel_set = self.teleporter_tensor.to(device)

        for t in range(T):
            inp = self.input_proj(x[:, t, :])  # [B, slot_dim]
            idx = ptr.view(B, 1, 1).expand(-1, 1, self.slot_dim)
            cur = state.gather(1, idx).squeeze(1)
            upd = self.gru(inp, cur)
            state.scatter_(1, idx, upd.unsqueeze(1))

            mask_tel = (ptr.unsqueeze(1) == tel_set.unsqueeze(0)).any(dim=1)
            p = torch.sigmoid(self.jump_score(upd)).squeeze(1)
            do_jump = mask_tel & (p > 0.8)
            if do_jump.any():
                rand_idx = torch.randint(len(tel_set), (do_jump.sum(),), device=device)
                jump_targets = tel_set[rand_idx]
                ptr = torch.where(do_jump, jump_targets, ptr)
                for j in rand_idx.tolist():
                    jump_hist_local[j] += 1
            ptr = (ptr + 1) % self.ring_len

        gather_idx = tel_set.view(1, -1, 1).expand(B, -1, self.slot_dim)
        tel_states = state.gather(1, gather_idx)
        tel_states = tel_states.view(B, -1)
        logits = self.head(tel_states)
        self.jump_hist = jump_hist_local.detach().cpu()
        return logits


class AudioM5(nn.Module):
    # From torchaudio tutorial (small CNN)
    def __init__(self, n_input=1, n_output=35, stride=4, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = torch.mean(x, dim=2)
        return self.fc1(x)


def _vision_transform():
    return T.Compose(
        [
            T.Resize((32, 32)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _vision_transform_color():
    return T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def get_vision_loader(name: str):
    try:
        if name == "mnist":
            dataset = torchvision.datasets.MNIST(
                root=os.path.join(DATA_DIR, "mnist"),
                train=True,
                download=not OFFLINE_ONLY,
                transform=_vision_transform(),
            )
            num_classes = 10
        elif name == "fashion_mnist":
            dataset = torchvision.datasets.FashionMNIST(
                root=os.path.join(DATA_DIR, "fashion_mnist"),
                train=True,
                download=not OFFLINE_ONLY,
                transform=_vision_transform(),
            )
            num_classes = 10
        elif name == "kmnist":
            dataset = torchvision.datasets.KMNIST(
                root=os.path.join(DATA_DIR, "kmnist"),
                train=True,
                download=not OFFLINE_ONLY,
                transform=_vision_transform(),
            )
            num_classes = 10
        elif name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(
                root=os.path.join(DATA_DIR, "cifar10"),
                train=True,
                download=not OFFLINE_ONLY,
                transform=_vision_transform_color(),
            )
            num_classes = 10
        elif name == "cifar100":
            dataset = torchvision.datasets.CIFAR100(
                root=os.path.join(DATA_DIR, "cifar100"),
                train=True,
                download=not OFFLINE_ONLY,
                transform=_vision_transform_color(),
            )
            num_classes = 100
        elif name == "svhn":
            dataset = torchvision.datasets.SVHN(
                root=os.path.join(DATA_DIR, "svhn"),
                split="train",
                download=not OFFLINE_ONLY,
                transform=_vision_transform_color(),
            )
            num_classes = 10
        else:
            return None, None
    except Exception as exc:
        log(f"Vision dataset {name} download/load failed: {exc}")
        return None, None

    subset_indices = list(range(min(MAX_SAMPLES, len(dataset))))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader, num_classes


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


class FileAudioDataset(torch.utils.data.Dataset):
    def __init__(self, items, num_classes, sample_rate=16000, max_len=16000):
        self.items = items
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.max_len = max_len

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
        return wav, label


def get_fsdd_loader():
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
        num_workers=2,
        pin_memory=True,
    )
    return loader, 10


def get_esc50_loader():
    root = os.path.join(DATA_DIR, "esc50")
    if OFFLINE_ONLY and not os.path.exists(root):
        raise RuntimeError("offline mode and esc50 not present")
    zip_url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    zip_path = _download_zip(zip_url, root, "esc50")
    extract_root = os.path.join(root, "ESC-50-master")
    if not os.path.exists(extract_root):
        _extract_zip(zip_path, root)
    meta_path = os.path.join(extract_root, "meta", "esc50.csv")
    audio_dir = os.path.join(extract_root, "audio")
    items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            label = int(row["target"])
            items.append((os.path.join(audio_dir, fname), label))
    items = items[:MAX_SAMPLES]
    dataset = FileAudioDataset(items, num_classes=50)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader, 50


def train_steps(model, loader, dataset_name, model_name, max_steps=MAX_STEPS):
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    losses = []
    start = time.time()
    step = 0

    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        elapsed = time.time() - start
        log(f"{dataset_name} | {model_name} | step {step:04d} | loss {loss.item():.4f} | t={elapsed:.1f}s")
        step += 1
        if step >= max_steps:
            break

    slope = compute_slope(losses)
    log(f"{dataset_name} | {model_name} | slope {slope:.6f} (loss/step over {len(losses)} steps)")
    return slope, losses


def build_eval_loader_from_subset(train_subset):
    eval_size = min(EVAL_SAMPLES, len(train_subset))
    if isinstance(train_subset, Subset):
        eval_indices = train_subset.indices[:eval_size]
        eval_subset = Subset(train_subset.dataset, eval_indices)
    else:
        eval_subset = Subset(train_subset, list(range(eval_size)))
    loader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return loader, eval_size


def eval_model(model, loader, dataset_name, model_name):
    model = model.to(DEVICE)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_seen += inputs.size(0)
    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    log(f"{dataset_name} | {model_name} | eval_loss {avg_loss:.4f} | eval_acc {acc:.4f} | eval_n {total_seen}")
    return {"eval_loss": avg_loss, "eval_acc": acc, "eval_n": total_seen}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    set_seed(SEED)
    log(f"Pilot Pulse start | device={DEVICE} | offline_only={OFFLINE_ONLY}")

    summary = []

    vision_names = ["mnist", "fashion_mnist", "kmnist", "cifar10"]
    for name in vision_names:
        log(f"Preparing vision dataset: {name}...")
        loader, num_classes = get_vision_loader(name)
        if loader is None:
            log(f"Skipping vision dataset {name}.")
            continue
        log(f"Vision {name} ready. classes={num_classes}")
        eval_loader, eval_n = build_eval_loader_from_subset(loader.dataset)

        vision_david = VisionDavidGRU(hidden=128, num_classes=num_classes)
        vision_swarm = VisionSwarmGRU(hidden=128, num_units=8, num_classes=num_classes)
        vision_specialist = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        vision_specialist.fc = nn.Linear(vision_specialist.fc.in_features, num_classes)

        log(f"Vision {name}: training David GRU (H=128)...")
        slope_d, _ = train_steps(vision_david, loader, f"vision/{name}", "david_gru_h128")
        eval_d = eval_model(vision_david, eval_loader, f"vision/{name}", "david_gru_h128")
        log(f"Vision {name}: training Swarm GRU (8xH=128)...")
        slope_sw, _ = train_steps(vision_swarm, loader, f"vision/{name}", "swarm_gru_8x128")
        eval_sw = eval_model(vision_swarm, eval_loader, f"vision/{name}", "swarm_gru_8x128")
        log(f"Vision {name}: training ResNet18 specialist...")
        slope_s, _ = train_steps(vision_specialist, loader, f"vision/{name}", "resnet18_specialist")
        eval_s = eval_model(vision_specialist, eval_loader, f"vision/{name}", "resnet18_specialist")
        summary.append(
            {
                "dataset": f"vision/{name}",
                "david_slope": slope_d,
                "swarm_slope": slope_sw,
                "specialist_slope": slope_s,
                "david_eval": eval_d,
                "swarm_eval": eval_sw,
                "specialist_eval": eval_s,
            }
        )

    audio_loaders = []
    try:
        audio_loaders.append(("fsdd",) + get_fsdd_loader())
    except Exception as exc:
        log(f"FSDD download/load failed: {exc}")
    try:
        audio_loaders.append(("esc50",) + get_esc50_loader())
    except Exception as exc:
        log(f"ESC-50 download/load failed: {exc}")

    for name, loader, num_labels in audio_loaders:
        log(f"Audio {name} ready. classes={num_labels}")
        eval_loader, eval_n = build_eval_loader_from_subset(loader.dataset)
        audio_david = AudioDavidGRU(hidden=128, num_classes=num_labels)
        audio_swarm = AudioSwarmGRU(hidden=128, num_units=8, num_classes=num_labels)
        audio_specialist = AudioM5(n_output=num_labels)
        log(f"Audio {name}: training David GRU (H=128)...")
        slope_d, _ = train_steps(audio_david, loader, f"audio/{name}", "david_gru_h128")
        eval_d = eval_model(audio_david, eval_loader, f"audio/{name}", "david_gru_h128")
        log(f"Audio {name}: training Swarm GRU (8xH=128)...")
        slope_sw, _ = train_steps(audio_swarm, loader, f"audio/{name}", "swarm_gru_8x128")
        eval_sw = eval_model(audio_swarm, eval_loader, f"audio/{name}", "swarm_gru_8x128")
        log(f"Audio {name}: training M5 specialist...")
        slope_s, _ = train_steps(audio_specialist, loader, f"audio/{name}", "m5_specialist")
        eval_s = eval_model(audio_specialist, eval_loader, f"audio/{name}", "m5_specialist")
        summary.append(
            {
                "dataset": f"audio/{name}",
                "david_slope": slope_d,
                "swarm_slope": slope_sw,
                "specialist_slope": slope_s,
                "david_eval": eval_d,
                "swarm_eval": eval_sw,
                "specialist_eval": eval_s,
            }
        )

    summary_path = os.path.join(ROOT, "pilot_pulse_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Pilot Pulse done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
