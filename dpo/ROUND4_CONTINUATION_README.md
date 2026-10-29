# Round 4 — Targeted Continuation vs Replay-Only (2026-10-29)

## Why this experiment?

Round3 reached **524/768 (68.23%)** on a fixed in-domain validation set and **103/256 (40.23%)** on a prior frozen OOD challenge set. The weakness is concentrated in three-digit multiplication (9/36) and three-digit `x_over_a` (4/39), while three-digit addition is already relatively strong (26/32).

Hypothesis: continuing the already-strong Round3 checkpoint with focused multiplication/inverse-multiplication examples improves math accuracy more than simply continuing on old data for the same number of optimizer steps. We retain the instructor's **original Course Loss** and the one-step `because` format.

## Controlled experiment

Both arms:

- Initialize **exactly** `dpo/experiments/round3_100k/dpo.pt` and hash-verify that checkpoint remains unchanged.
- Use 100,000 training rows, 2 epochs, batch size 64 (1562 full batches/epoch; 32 leftover rows per epoch are skipped by teacher's original `get_batches`).
- Learning rate = 2e-5 (new AdamW optimizer/scheduler for both arms), seed 42; same architecture, original full-sequence Course Loss.
- Use the same 70,000 OLD training rows as common replay base.

The remaining 30% differs:

| Arm | Shared older examples | Remaining 30% | Why |
|---|---:|---|---|
| Replay-only control | 70,000 | 30,000 other Round3 training rows | Extra epochs alone |
| Treatment | 70,000 | 15,000 **hard older training** examples + 15,000 **new** product/inverse facts, product in 401–999 | Targeted arithmetic practice |

The 15,000 hard older rows preferentially sample multiplication and inverse categories with challenging answer sizes; the new 15,000 cover `multiply`, `divide`, `x_times_a`, `a_times_x`, `x_over_a`, `a_over_x` with verified one-step explanations. The source of all Round3 replay examples is Round3 **train**, never val or test. The extension only uses canonical facts with `stable_split(key, 42) == train`, excluding the older frozen 256-val facts and reserved course examples.

**Important limitation:** This is a *combined targeted-curriculum* test (hard old oversampling + expanded new facts), not a one-variable decomposition of either component. If effective, separate them in a later ablation.

No new SFT, no custom Loss, no rewrites of the existing Round3 checkpoint, no use of sealed Round3 final Test for model selection. Final Test labels are only accessed by the data generator for mathematically fact-key contamination prevention, not scored.

## Phase 1 — generate and inspect on TC1 (CPU only)

```bash
cd ~/NanoGPT-Math
git status --short
git pull --ff-only origin main
source .venv/bin/activate

python scripts/build_round4_data.py --profile mini --verify-only
python scripts/build_round4_data.py --profile full --verify-only
python scripts/build_round4_data.py --profile full

python scripts/train_round4_matched.py --arm treatment --check-only
python scripts/train_round4_matched.py --arm replay_only --check-only
```

Output: `dpo/data/round4_matched_full/{treatment_pairs.json,replay_only_pairs.json,manifest.json}`.
Every source/arm file is hashed. Existing output folders are never overwritten. No GPU required for these commands. If the data folder already exists, inspect it instead of deleting or recreating it.

## Phase 2 — small GPU smoke (one GPU)

```bash
sbatch scripts/round4_smoke_treatment.sbatch
squeue -u "$USER"
```

Smoke uses only 128 rows from the treatment data, 1 epoch, 16 batch size; it only confirms checkpoint loading, Loss, output writing, and scorer. It is **not** a meaningful math improvement trial. Log: `r4_smoke_JOBID.out`, artifacts: `dpo/experiments/round4_smoke_treatment/`.

## Phase 3 — submit matched full experiments (one GPU per job)

Only after smoke passes:

```bash
sbatch scripts/round4_continue_treatment.sbatch
sbatch scripts/round4_continue_replay.sbatch
squeue -u "$USER"
```

TC1 limits coursework users to one GPU concurrently; both jobs can be submitted but normally execute sequentially. Each needs one V100. The `r4_target_JOBID.out` and `r4_replay_JOBID.out` files show batch-level progress every 250 batches.

Separate checkpoints/metrics:

- `dpo/experiments/round4_treatment/dpo_epoch1.pt`, `dpo_epoch2.pt`, `summary.json`
- `dpo/experiments/round4_replay_only/dpo_epoch1.pt`, `dpo_epoch2.pt`, `summary.json`

Each epoch automatically evaluates **the same** 256 Round3 train-probe examples, 768 in-domain Round3 validation cases and 256 older OOD validation cases. No Round3 final test is evaluated.

After BOTH jobs finish:

```bash
python scripts/compare_round4_matched.py
```

Look first at treatment-minus-replay on the fixed 768 questions, and at its impact on multiplication, `x_over_a`, addition, and the older OOD set. Watch for catastrophic forgetting.

## Interpretation

- Treatment > replay with retained addition accuracy: evidence that the targeted curriculum helps **beyond continued training alone**.
- Treatment = replay but both > Round3: additional epochs, not targeting, may explain gains.
- Treatment improves multiplication while harming addition/overall: examine replay ratio, not Loss coefficients.
- No arm improves: reconsider curriculum and arithmetic representation before heavier Loss engineering.

This repository contains code **only**. Generated datasets live on TC1 and are not automatically committed. No new Round4 model has been trained by these commits.
