# Round 6 — Four Output Formats, Fair SFT Initialization

**Current status, 2026-10-30:** all FOUR arms completed E1/E2. The user has now elected to STOP B (`answer_only`) and C (`decomposition_after`) at E2, preserving their existing results and checkpoints, and to CONTINUE ONLY A (`original`) and D (`decomposition_before`) to E5. Old Round3/4/5 experiments remain immutable. The original four-arm protocol below documents E1/E2, but the active follow-up is the two-arm A/D continuation described immediately below.

## ACTIVE: A/D-only continuation after Epoch 2

**Already completed E2:** A Val180/768, Extended84/384, OOD24/256 and Direct16/128; D Val174/768, Extended58/384, OOD14/256 and Direct3/128, all under the same strict historical answer-first score. D's separate exploratory final-answer 79/384 (24/128 direct) does NOT replace its primary 58/384 (3/128 direct); actual TA extraction remains unknown. D also produced 17 fully verified arithmetic chains on E2, compared with 0 at E1.

Update from GitHub:

```bash
cd ~/NanoGPT-Math
git pull --ff-only origin main
source .venv/bin/activate
python scripts/compare_round6_ad.py --self-test
python scripts/compare_round6_ad.py
```

**Submit only two E3 jobs (if not already submitted or running):**

```bash
squeue -u "$USER"
sbatch scripts/round6_epoch.sbatch original 3
sbatch scripts/round6_epoch.sbatch decomposition_before 3
```

TC1 allows **at most two submitted jobs per user**, only **one GPU running**. Never use the old `submit_round6_arm.sh` for multi-epoch queuing here; it attempts 3–5 jobs and will exceed QoS limits. Once BOTH E3 jobs have completed successfully (check `epoch_summary.json` and `sacct`), inspect:

```bash
python scripts/compare_round6_ad.py
python scripts/audit_round6_reasoning_outputs.py --epoch 3
```

Repeat exactly two jobs for E4, then E5, only after the preceding epoch's summary exists in BOTH arms:

```bash
sbatch scripts/round6_epoch.sbatch original 4
sbatch scripts/round6_epoch.sbatch decomposition_before 4
# After both E4 jobs finish and produce summaries:
sbatch scripts/round6_epoch.sbatch original 5
sbatch scripts/round6_epoch.sbatch decomposition_before 5
```

**Main comparison:** strict fixed `The answer is <integer>`-at-start scorer applied IDENTICALLY to both A and D on Val768, OOD256, Extended384 and direct-product128. Run `python scripts/compare_round6_ad.py --diagnostics` only for explicitly SEPARATE exploratory metrics on D (final answer after reasoning, correct arithmetic steps). No change to checkpoints, Loss, dataset, or sealed Test. A is the better course-grading candidate so far, D is the investigation arm. A-vs-D bundles decomposition position and content; without continuing C one cannot causally isolate positioning.

---

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
- **ONE primary score for all four arms**: `strict_accuracy` with response beginning `The answer is <integer>`. This is the existing historical Round3/4/5 answer-extraction convention; the same scorer is used for A/B/C/D and remains unchanged throughout Round6.
- `final_accuracy` (recognizing a complete `<steps> so The answer is N.` completion), `reason_before_answer`, and `valid_calculation` are **separate, purely exploratory diagnostics**, never substituted for primary model selection. Inspect via `python scripts/compare_round6_from_sft.py --diagnostics`; ordinary `python scripts/compare_round6_from_sft.py` reports only uniform primary metrics.
- The instructor Notebook's Step 8 shows generating test answers but does **not** disclose the actual TA scoring/extraction code. Our historical strict evaluator is thus NOT a confirmed imitation of the hidden TA parser. If the TA uses an anywhere-in-output search for `The answer is N`, reasoning-first might be accepted, but this has NOT been verified. Do not select D as a course submission winner based only on secondary exploratory diagnostics.

### Hypotheses and decision criteria

**H1 B vs A:** does removing `because` improve the first numeric answer and original course-format accuracy? Both answer-first and suitable for existing grader.

**H2 D vs C:** can pre-answer steps improve performance under the SAME fixed primary answer-extraction scoring, relative to the 16,271 matched post-answer decompositions? Examine `final_accuracy`, `reason_before_answer`, `valid_calculation` and format failures only as supplemental diagnostics; never mix these with the primary rank.

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

## Historical first-round procedure (E1 already complete for all four; do NOT rerun)

```bash
sbatch scripts/round6_epoch.sbatch original 1
sbatch scripts/round6_epoch.sbatch answer_only 1
sbatch scripts/round6_epoch.sbatch decomposition_after 1
sbatch scripts/round6_epoch.sbatch decomposition_before 1
squeue -u "$USER"
```

Historically, after all four Epoch-1 jobs were COMPLETE, inspect `python scripts/compare_round6_from_sft.py`. Confirm that the new `decomposition_before` arm actually emits `reason_before_answer` completions and their final arithmetic is readable; a near-zero rate may be a sign of format inertia even from SFT, not necessarily an arithmetic-learning failure.

## Historical four-arm follow-up plan (superseded by ACTIVE A/D section above)

```bash
bash scripts/submit_round6_arm.sh original 2
bash scripts/submit_round6_arm.sh answer_only 2
bash scripts/submit_round6_arm.sh decomposition_after 2
bash scripts/submit_round6_arm.sh decomposition_before 2
squeue -u "$USER"
```

**TC1 is 1 GPU/user:** four arms may queue, not run simultaneously. Each command submits the remaining `afterok`-dependent epoch jobs for one arm, and verifies the immediately preceding epoch was fully evaluated. For a full run without a staged pilot, omit the final `2` to submit epochs 1–5. Avoid submitting all 20 jobs if the cluster has queued-job limits. `#SBATCH --time=04:00:00` is a per-epoch walltime request, not a guaranteed time estimate. Each epoch has its own nonoverwriting directory. If one epoch fails, later epochs will NOT run automatically.

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

It rejects unmatched LR, seed, data, model origin, scoring budgets, epoch settings or question order. The default report ranks A/B/C/D exclusively using the SAME historical strict parser for every arm and every split. For optional, clearly labeled research diagnostics, append `--diagnostics`.

Final model files will be:
```text
dpo/experiments/round6_from_sft/original/dpo.pt
dpo/experiments/round6_from_sft/answer_only/dpo.pt
dpo/experiments/round6_from_sft/decomposition_after/dpo.pt
dpo/experiments/round6_from_sft/decomposition_before/dpo.pt
```

Do **not** open the sealed final Test to choose between these; select by pre-stated target and format constraints, then assess the chosen checkpoint once. Historical Round3/4/5 numbers are contextual, not clean causal comparators.

**Implementation:** `scripts/build_round6_formats.py`, `scripts/evaluate_round6_formats.py`, `scripts/train_round6_from_sft.py`, `scripts/compare_round6_from_sft.py`, `scripts/round6_smoke.sbatch`, `scripts/round6_epoch.sbatch`, `scripts/submit_round6_arm.sh`.
