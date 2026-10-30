# Round8: Model-Caused Error Mining + Bounded Number Variants

**Status:** GitHub code committed, **no Round8 GPU mining or training run yet**.

This explicitly replaces the **unsuccessful Round7 policy of adding products 1000–4999**. User's request: first make the model answer the EXISTING difficulty of arithmetic correctly, use the CURRENT MODEL'S OWN TRAIN-only WRONG COMPLETIONS as Rejected, and create **small numerical variations** of those training mistakes without touching held-out/public benchmark math facts.

## Why this experiment is different

Freeze **Round7 Replay** as current baseline and mining model (strict Val567/768, old OOD109/256, Extended174/384, Direct28/128). Use its actual deterministic NanoGPT `model.py` output, not arbitrary wrong-number corruption. For a TRAIN question `47*13=?` with gold611, e.g. a genuine model generated response `The answer is 601 because 47*13 equals 601.` becomes Rejected in the experimental arm. Only if THIS COMPLETION IS ACTUALLY GENERATED AND INCORRECT; the example is not claimed to have been observed.

It mines a deterministic stratified sample (120 unique TRAIN prompts per each of 12 math classes, max1440), retaining only model-generated WRONG **complete answer-first** statements with a numeric error and `because ... .` output. No format-only failures or correct final numbers are treated as hard negatives.

For EACH actual error, derive nearby `a±1..3`, `b±1..3` variations of the original TRAIN question while retaining its question type. Multiplication results are constrained to **101..999**, **never >=1000**. Additions stay bounded. Variations are *new exercises*, not necessarily previously-unseen mathematical facts: re-practicing an existing TRAIN fact is acceptable. Every variant:
- belongs to the original `stable_split(fact_key,42)=='train'`
- must change the mathematical fact of the source wrong TRAIN exercise (even though it may overlap OTHER training facts)
- shares NO mathematical fact with any of fixed Val768/Old OOD256/Extended384
- shares NO mathematical fact with 61+41 public-year benchmark questions (these are loaded ONLY as an exclusion list, never as seeds)
- is generated from original TRAIN question and verified via `parse_relation`

The frozen model then also generates completions for the variants. Only variants actually answered wrong by that model are retained as new hard-negative examples. We also report how many changed-number facts were already present in other source TRAIN rows. **The sealed Round3 independent Test file is never opened.**

## Crucial controlled experiment

Create two identical 100,000-row training datasets, identical prompts, Chosen positive strings, mathematical facts, shuffle/order and source Replay rows. The **only difference** is the negative completion on mined-error rows:

- `control`: normal instructor-style `Sorry, I don't know.` Rejected;
- `hardneg`: the actual base model's wrong complete answer as Rejected.

Old training Replay fills remaining rows. Errors and variants can each occupy **up to 5000 training-row exposures**, but each unique mined question may repeat **no more than five times**. If only a small number of wrong examples are mined, the hard-negative rows occupy a proportionately small share of the 100k dataset; do not invent/Re-label wrong samples to force a fixed quota. Same `Round7 Replay` initialization, Course Loss, 1 epoch, LR1e-5, Batch64, seed42, same 1562 steps, same evaluator across both arms.

This isolates the **added benefit of model-generated hard Rejected vs refusal**, holding difficulty targeting and number-variation Chosen examples constant. It does **not** independently isolate augmentation from continuing regular Replay: base Round7 Replay is a historical reference, not a matched extra third arm. Because the course Loss uses *full-sequence mean log-probabilities*, differing Rejected lengths/format may influence this experiment; report this as a methodological limitation and consider completion-only loss only as a separate follow-up controlled experiment, not mix it in this one.

## Step 1 — CPU self-test, then ONE GPU mining job

```bash
cd ~/NanoGPT-Math
git pull --ff-only origin main
source .venv/bin/activate
python scripts/mine_round8_train_hard_negatives.py --self-test-only
squeue -u "$USER"
sbatch scripts/round8_mining.sbatch
```

TC1 user QoS permits at most two submitted jobs and ONE active GPU. Mining can take significant time because NanoGPT greedy decoding is character autoregressive. `round8_mining.sbatch` uses **one GPU with a 4-hour job limit**. Do not submit training in parallel with mining, because it depends on the mined artifacts.

When the miner exits successfully, inspect:
```bash
cat dpo/experiments/round8_hard_negative_mining_seed42/manifest.json
python - <<'PY'
import json
from pathlib import Path
p=Path("dpo/experiments/round8_hard_negative_mining_seed42")
for name in ("original_hard_negatives.json","variant_hard_negatives.json"):
    rows=json.loads((p/name).read_text())
    print(name, "N=",len(rows))
    for r in rows[:3]:
        print("  question:",r["prompt"],"GOLD:",r["answer"])
        print("  Chosen:",r["positive"])
        print("  Rejected:",r["negative"])
PY
```

**Do not launch training before checking the actual wrong-question counts and examples.** Default mining is a pilot; if sparse (fewer than 20 original /10 variant errors) the paired builder intentionally refuses to continue.

## Step 2 — CPU generate matched same-positive data

```bash
python scripts/build_round8_hardneg_data.py --verify-only
python scripts/build_round8_hardneg_data.py
python scripts/train_round8_matched.py --arm control --check-only
python scripts/train_round8_matched.py --arm hardneg --check-only
```

Data saved under `dpo/data/round8_hardneg_full` with manifest SHA, counts, repeat cap and unmodified positive paired order. All original/source protection hashes checked. No GPUs needed. Generation refuses to overwrite existing output.

## Step 3 — TWO matched GPU jobs (ONLY AFTER mining, CPU audit and user review)

```bash
squeue -u "$USER"
sbatch scripts/round8_train.sbatch control
sbatch scripts/round8_train.sbatch hardneg
```

Separate new output paths `dpo/experiments/round8_control/` and `dpo/experiments/round8_hardneg/`. **Do NOT overwrite Round7 Replay, Round5-A or any old checkpoint.** Evaluate strict historic answer-first on fixed Val768, OOD256, Extended384, Probe256. No public Blind61/41 and no sealed independent Test accessed.

After both succeed:
```bash
python scripts/compare_round8_hardneg.py
```

The primary question is **HARDNEG minus CONTROL** on paired same questions. Read categories/direct-product128 and product bands from the printed report. The actual TA grader remains unknown; do not claim official performance.

## Files

- `scripts/mine_round8_train_hard_negatives.py` — model-generated training errors and nearby bounded train-only variants
- `scripts/round8_mining.sbatch` — single GPU mining
- `scripts/build_round8_hardneg_data.py` — same-Chosen paired preference dataset
- `scripts/train_round8_matched.py` — one epoch original instructor Course Loss from Round7 Replay, no overwrites
- `scripts/round8_train.sbatch` — single GPU per training arm
- `scripts/compare_round8_hardneg.py` — read-only paired accuracy comparison

All work in this round must be transparently described in the final assignment notebook. Revisit course Task4 explanation and provenance of mined errors, synthetic variants and controls if including Round8 in the report.
