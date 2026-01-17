# Small Synthetic Bench (PRIME C-19)

Command:

```bash
python tools/bench_small_prime.py
```

Defaults used (script defaults):

- epochs: 120
- batch_size: 64
- lr: 1e-3
- ring_len: 128
- slot_dim: 16
- gauss_k: 2
- gauss_tau: 2.0
- seq_mode: steps2 (x,y as 2-step sequence with input_dim=1)
- activations: c19, relu, silu

Pointer/loop safety (set inside script):

- SATIETY_THRESH=1.1 (disabled)
- STATE_DECAY=0.99
- STATE_CLIP=2.0
- PTR_JUMP_DISABLED=1
- PTR_WALK_PROB=0.10
- ptr_update_every=1

Dataset generators are matched to `TrainingBenchmark.cs` in `G:\1. GPT Workfolder\C3AiLab.Desktop`:

- XOR: 512 samples, noise=0.2
- Two Moons: 768 samples, noise=0.12
- Circles: 768 samples, noise=0.1
- Spiral: 900 samples, classes=3, noise=0.18 (phase noise)
- Sine Regression: 512 samples, noise=0.05

Results (eval set, 80/20 split):

```
xor        | c19    | loss=0.0326 acc=1.0000
xor        | relu   | loss=0.0334 acc=1.0000
xor        | silu   | loss=0.0448 acc=1.0000
two_moons  | c19    | loss=0.0075 acc=1.0000
two_moons  | relu   | loss=0.0082 acc=1.0000
two_moons  | silu   | loss=0.0065 acc=1.0000
circles    | c19    | loss=0.0308 acc=0.9935
circles    | relu   | loss=0.0311 acc=0.9870
circles    | silu   | loss=0.0445 acc=0.9935
spiral     | c19    | loss=0.1798 acc=0.9667
spiral     | relu   | loss=0.2881 acc=0.8833
spiral     | silu   | loss=0.2398 acc=0.9111
sine       | c19    | mse=0.0078
sine       | relu   | mse=0.0367
sine       | silu   | mse=0.0203
```
