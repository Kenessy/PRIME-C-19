import subprocess
import json
import os

base_env = {
    "TP6_SYNTH":"1",
    "TP6_SYNTH_MODE":"assoc_clean",
    "TP6_SYNTH_LEN":"32",
    "TP6_ASSOC_KEYS":"4",
    "TP6_ASSOC_PAIRS":"2",
    "TP6_MAX_SAMPLES":"1024",
    "TP6_BATCH_SIZE":"32",
    "TP6_MAX_STEPS":"400",
    "TP6_PTR_SOFT_GATE":"1",
    "TP6_PTR_WALK_PROB":"0.05",
    "TP6_PTR_INERTIA":"0.1",
    "TP6_PTR_DEADZONE":"0",
    "TP6_PTR_NO_ROUND":"1",
    "TP6_SOFT_READOUT":"1",
    "TP6_LMOVE":"0",
    "TP6_PANIC_ENABLED":"0",
    "TP6_THERMO_ENABLED":"0",
    "TP6_PTR_UPDATE_EVERY":"8",
}

activations = ["c19", "silu"]

results = []
for act in activations:
    env = os.environ.copy()
    env.update(base_env)
    env["TP6_ACT"] = act
    print(f"\n=== HARD ASSOC_CLEAN ACT={act} ===")
    proc = subprocess.run(
        ["python", "tournament_phase6.py"],
        cwd=r"G:\\AI\\pilot_pulse",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout
    eval_acc = None
    eval_loss = None
    for line in out.splitlines():
        if "eval_acc" in line and "eval_loss" in line:
            parts = line.strip().split()
            for i,p in enumerate(parts):
                if p == "eval_loss":
                    eval_loss = float(parts[i+1])
                if p == "eval_acc":
                    eval_acc = float(parts[i+1])
    results.append({"act": act, "eval_loss": eval_loss, "eval_acc": eval_acc, "ok": proc.returncode==0})
    print(f"{act}: eval_loss={eval_loss} eval_acc={eval_acc} ok={proc.returncode==0}")

print("\n=== SUMMARY ===")
print(json.dumps(results, indent=2))
