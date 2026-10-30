# Round 6 — Four Output Formats, Fair SFT Initialization

**Current status:** code committed. Data creation, TC1 CPU checks, GPU smoke, and all training are still to be run by the user. Old Round3/4/5 experiments are kept immutable.

## Why start over at SFT?

Round4 Treatment E1 already learned the answer-first `because` template. Asking it to switch to reasoning-first with one small-LR epoch would test *format adaptation* rather than which training format works better. Instead, **all four Round6 arms begin from the identical `sft/gpt.pt`**, no existing DPO weights, and use the identical 100,000 math questions, labels, refusal negatives and row order. This is a matched answer-format experiment, not a matched token-cost experiment; outputs have different lengths.

The underlying 100k dataset is **Round4 Treatment**: 70k previous Round3 training rows + 15k harder previous multiplication/inverse rows + 15k new 401–999 product/inverse rows, with the same existing mathematical-fact-disjoint fixed validation sets.

| Arm | Positive formatting | Rows affected relative to A |
|---|---|---:|
| **original (A)** | `23*17=? The answer is 391 because 23*17 equals 391.` | control 0 |
| **answer_only (B)** | `23*17=? The answer is 391.` | all 100,000 |
| **decomposition_after (C)** | `23*17=? The answer is 391 because 23*10=230 and 23*7=161 and 230+161=391.` | 16,271 |
| **decomposition_before (D)** | `23*17=? 23*10=230 and 23*7=161 and 230+161=391 so The answer is 391.` | same 16,271 |

The **D arm's other 83,729 rows are identical to C** (which are, in turn, identical to A for these rows). Counts 16,271 = 8,486 multiply + 7,785 x_over_a are inherited from the verified Round5 source manifest and independently checked during Round6 data generation. Rows may repeat mathematical prompts, so these are training **records**, not necessarily distinct questions. The model stops at first `.` or newline, so no interior periods/newlines in D's decomposed positives.

## Four-arm protocol

- Initialize model weights from `sft/gpt.pt` (original ~8.84M-parameter NanoGPT), exactly once at Epoch 1 per arm; restore that arm's own previous state in Epochs 2–5.
- **Original instructor Course Loss**, full-sequence mean preference objective with chosen bonus. This is NOT standard reference-model DPO; no new Loss in Round6.
- **100,000 rows per epoch** per arm (last 32 skipped; 1,562 full batches), batch 64, seed 42.
- Base LR `1e-4`, AdamW weight decay 0.01, original five-epoch CosineAnnealingLR to 10% of initial LR.
- **Five complete epochs** per arm, with 1 GPU job / epoch, chained using Slurm `afterok`. At each epoch checkpoint preserve model weights, AdamW moments, LR scheduler and Torch CPU/CUDA RNG. The sample order is a deterministic SHA256-verifiable permutation of the same 100k indices each epoch across all arms (not reused Round3's historical shuffling).
- Every epoch evaluates the same fixed **Round3 Val768**, **Old OOD256**, **Round4 Extended Val384**, and **Train Probe256** (the last may contain repeated prompts). The sealed **Round3 Test768 is never loaded, evaluated, or used for model selection**.
- All four arms evaluated with the SAME period/newline greedy stop and `max_new_tokens=160`. This differs from Round3/4/5's earlier 80-character limit; historical comparisons remain descriptive unless re-evaluated under identical limits.
- Every evaluation reports two numerical metrics:
  1. **strict_accuracy**: model's response begins `The answer is <integer>`; identical first-answer scoring protocol to previous Round3/4/5 saved results.
  2. **final_accuracy**: additionally accepts complete, recognizable `<validly patterned steps> so The answer is N.` after the steps. Mathematical correctness of steps is scored separately as `valid_calculation`; the final number can be correct even when arithmetic steps are wrong, so use both.
- The strict metric is **NOT** a confirmed imitation of the lecturer/TA hidden evaluation code. A D output that is correct under final-answer metric may be unusable for the course grader. Do not select D as a submission checkpoint without verifying required answer format and end-to-end validation.

### Hypotheses and decision criteria

**H1 B vs A:** does removing `because` improve the first numeric answer and original course-format accuracy? Both answer-first and suitable for existing grader.

**H2 D vs C:** can pre-answer steps improve final numerical correctness **relative to the same 16,271 decomposition records placed after the answer**? Compare `final_accuracy`, `strict_accuracy`, `reason_before_answer`, `valid_calculation` and format failures.

**Primary:** the **128 fixed direct large-product** questions within Extended Val384 (64 multiply and 64 x_over_a). Secondary: entire Extended Val384; guardrails: Val768 and Old OOD256. Use matched per-item-only correctness comparisons, not merely different datasets' average accuracy.

Caveats: the 4 arms' token lengths/whole-sequence Course Loss denominators differ, so this is a controlled **format intervention** rather than a pure chain-of-thought causal mechanism. If B wins, a new Loss ablation is sensible; if D wins on flexible final-answer scoring but fails strict formatting, it is an exploratory model rather than a course-ready winner. Further random data expansion is lower priority until these questions are settled.

## CPU-only phase on TC1 — do this FIRST

```bash
cd ~/NanoGPT-Math
git status --short
git pull --ff-only origin main
source .venv/bin/activate
python scripts/evaluate_round6_formats.py
python scripts/build_round6_formats.py --verify-only
python scripts/build_round6_formats.py

# Verify all four datasets and frozen val hashes before any GPU:
for arm in original answer_only decomposition_after decomposition_before; do
  python scripts/train_round6_from_sft.py --arm "$arm" --epoch 1 --check-only
done
```

Expected: 100,000 each arm, 100,000 answer-only rows, 16,271 rows with reasoning moved ahead of answer. Generator refuses overwriting `dpo/data/round6_from_sft/`, checks math and SHA256, tokenizer vocabulary and 256-character block size.

## GPU smoke

```bash
sbatch scripts/round6_smoke.sbatch
squeue -u "$USER"
```

Read the resulting `round6_smoke_*.out` and `.err` to confirm full success. Smoke writes only to `dpo/experiments/round6_from_sft/_smoke_decomposition_before/epoch01/`.

## Full 4 x 5 epochs: submit per arm

```bash
bash scripts/submit_round6_arm.sh original
bash scripts/submit_round6_arm.sh answer_only
bash scripts/submit_round6_arm.sh decomposition_after
bash scripts/submit_round6_arm.sh decomposition_before
squeue -u "$USER"
```

**TC1 is 1 GPU/user:** four arms may queue, not run simultaneously. Each command submits five `afterok`-dependent epoch jobs for one arm. Do not submit 20 jobs if there is a cluster queued-job limit; submit one arm at a time as allowed. `#SBATCH --time=04:00:00` is a per-epoch walltime request, not a guaranteed time estimate. Each epoch has its own nonoverwriting directory. If one epoch fails, later epochs will NOT run automatically.

For manual single job:

```bash
sbatch scripts/round6_epoch.sbatch answer_only 1
```

For incomplete evaluation recovery **when the training_state.pt was already saved**:

```bash
python scripts/train_round6_from_sft.py --arm answer_only --epoch 1 --eval-only \
  --device cuda --lr 1e-4 --batch-size 64 --max-new-tokens 160
```

Run that only inside a GPU allocation (interactive or adapted Slurm script); `--eval-only` does not rerun training. The original partial output directory must exist.

## Epoch-curve and comparison script

After each full set of four arm epochs (or when any matching epochs are available):

```bash
python scripts/compare_round6_from_sft.py
```

It rejects unmatched LR, seed, data, model origin, scoring budgets, epoch settings or question order. Prints A/B and C/D paired scores for each fixed validation split and direct-large-product target subset.

Final model files will be:
```text
dpo/experiments/round6_from_sft/original/dpo.pt
dpo/experiments/round6_from_sft/answer_only/dpo.pt
dpo/experiments/round6_from_sft/decomposition_after/dpo.pt
dpo/experiments/round6_from_sft/decomposition_before/dpo.pt
```

Do **not** open the sealed final Test to choose between these; select by pre-stated target and format constraints, then assess the chosen checkpoint once. Historical Round3/4/5 numbers are contextual, not clean causal comparators.

**Implementation:** `scripts/build_round6_formats.py`, `scripts/evaluate_round6_formats.py`, `scripts/train_round6_from_sft.py`, `scripts/compare_round6_from_sft.py`, `scripts/round6_smoke.sbatch`, `scripts/round6_epoch.sbatch`, `scripts/submit_round6_arm.sh`.
