# RUISS Activation Benchmark (Notes)

RUISS is a relative activation score computed against a ReLU baseline:

```
ratio = baselineCost / totalCost
RUISS = 100 * ratio / (1 + ratio)
```

Higher is better.

## Recorded Results (RUISS-Brute v1)

- Activation: C-13
  - RUISS score: 59.339511827521974
  - SuiteLabel: RUISS-Brute v1

- Activation: ReLU (baseline)
  - RUISS score: 50.0
  - SuiteLabel: RUISS-Brute v1

Source (local):
- G:\1. GPT Workfolder\C3AiLab-App\data\benchmarks\ruiss-runs.json
- G:\1. GPT Workfolder\C3AiLab.Desktop\RuissEvaluator.cs

Note: C-19 has not been scored in this suite yet; C-13 is the closest precursor.
