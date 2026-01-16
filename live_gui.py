import argparse
import json
import os
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def tail_jsonl(path, state):
    if not os.path.exists(path):
        return []
    size = os.path.getsize(path)
    if size < state["pos"]:
        state["pos"] = 0
    with open(path, "r", encoding="utf-8") as handle:
        handle.seek(state["pos"])
        lines = handle.readlines()
        state["pos"] = handle.tell()
    return lines


def main():
    parser = argparse.ArgumentParser(description="Live GUI for Project David metrics.")
    parser.add_argument("--path", default=os.path.join("traces", "current", "live_trace.json"), help="JSONL trace path")
    parser.add_argument("--interval", type=float, default=5.0, help="Update interval (sec)")
    parser.add_argument("--max-points", type=int, default=300, help="Max points to keep")
    args = parser.parse_args()

    steps = deque(maxlen=args.max_points)
    losses = deque(maxlen=args.max_points)
    grad_norms = deque(maxlen=args.max_points)
    flip_rates = deque(maxlen=args.max_points)
    mean_dwells = deque(maxlen=args.max_points)

    ptr_entropy = None
    state_entropy = None
    state_flip = None
    state_abab = None
    last_time = None

    file_state = {"pos": 0}

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    ax_loss = axs[0, 0]
    ax_grad = axs[0, 1]
    ax_flip = axs[1, 0]
    ax_text = axs[1, 1]
    ax_text.axis("off")

    loss_line, = ax_loss.plot([], [], color="tab:blue", lw=1.5)
    grad_line, = ax_grad.plot([], [], color="tab:orange", lw=1.5)
    flip_line, = ax_flip.plot([], [], color="tab:red", lw=1.5, label="flip")
    dwell_line, = ax_flip.plot([], [], color="tab:green", lw=1.5, label="dwell")
    ax_flip.legend(loc="upper right", fontsize=8)

    ax_loss.set_title("Loss")
    ax_grad.set_title("Grad Norm (theta_ptr)")
    ax_flip.set_title("Flip Rate / Mean Dwell")
    ax_loss.set_xlabel("step")
    ax_grad.set_xlabel("step")
    ax_flip.set_xlabel("step")

    text_box = ax_text.text(0.02, 0.98, "", va="top", ha="left", family="monospace")

    def update(_frame):
        nonlocal ptr_entropy, state_entropy, state_flip, state_abab, last_time
        lines = tail_jsonl(args.path, file_state)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            if step is None:
                continue
            steps.append(step)
            losses.append(row.get("loss"))
            grad_norms.append(row.get("grad_norm_theta_ptr"))
            flip_rates.append(row.get("ptr_flip_rate"))
            mean_dwells.append(row.get("ptr_mean_dwell"))
            ptr_entropy = row.get("pointer_entropy")
            state_entropy = row.get("state_loop_entropy")
            state_flip = row.get("state_loop_flip_rate")
            state_abab = row.get("state_loop_abab_rate")
            last_time = row.get("time_sec")

        if not steps:
            return

        loss_line.set_data(steps, losses)
        grad_line.set_data(steps, grad_norms)
        flip_line.set_data(steps, flip_rates)
        dwell_line.set_data(steps, mean_dwells)

        for ax in (ax_loss, ax_grad, ax_flip):
            ax.relim()
            ax.autoscale_view()

        summary = [
            f"path: {args.path}",
            f"step: {steps[-1]}",
            f"time_sec: {last_time:.3f}" if last_time is not None else "time_sec: n/a",
            f"loss: {losses[-1]:.4f}" if losses[-1] is not None else "loss: n/a",
            f"grad_norm: {grad_norms[-1]:.3e}" if grad_norms[-1] is not None else "grad_norm: n/a",
            f"flip_rate: {flip_rates[-1]:.3f}" if flip_rates[-1] is not None else "flip_rate: n/a",
            f"mean_dwell: {mean_dwells[-1]:.2f}" if mean_dwells[-1] is not None else "mean_dwell: n/a",
        ]
        if ptr_entropy is not None:
            summary.append(f"ptr_entropy: {ptr_entropy:.3f}")
        if state_entropy is not None:
            summary.append(f"state_entropy: {state_entropy:.3f}")
        if state_flip is not None:
            summary.append(f"state_flip: {state_flip:.3f}")
        if state_abab is not None:
            summary.append(f"state_abab: {state_abab:.3f}")

        text_box.set_text("\n".join(summary))

    FuncAnimation(fig, update, interval=int(args.interval * 1000))
    fig.suptitle("Project David Live Monitor", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
