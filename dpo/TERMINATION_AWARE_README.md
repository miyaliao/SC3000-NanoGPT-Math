# SC3000 NanoGPT-Math — Overfit-256 Termination-Aware Loss

## Hypothesis

The previous **Answer-Focused Loss** overfit experiment ranked chosen answers above rejected answers in **255/256** pairs but obtained **0/256** exact greedy accuracy on seen training questions. Its natural completions rarely/never terminated with the course's period (`.`) token. Here we isolate **one change**: an extra teacher-forced NLL for the period immediately **after the correct integer answer**.

We do **not** claim it will solve mathematical correctness: in the previous experiment only 26/256 generated number strings began with the entire correct answer; resolving termination alone cannot fix wrong first digits or arithmetic dependencies.

## Experiment specification

- **Data:** Exactly the same 256 Round 1C incorrect-answer pairs sampled with `seed=42` by `overfit256_course.select_rows`; same frozen 256 integer validation questions, held out from training.
- **Initialization:** `sft/gpt.pt` (not continued from earlier DPO checkpoints).
- **Model:** No modifications to `model.py`; TA-like greedy period/newline stopping for evaluation.
- **Training:** 128 epochs, 64 batch, 512 steps, unchanged `beta=0.5`, AdamW, learning rate scheduler, gradient clipping.
- **Previous Answer-Focused objective:**

  `L_old = L_preference_completion_sum + 1.0 * NLL_chosen_numeric_mean + 0.25 * NLL_chosen_response_mean`

- **Only change:**

  `L_new = L_old + gamma * [-log Pθ('.' | question, ' The answer is ', correct integer)]`

  `gamma = 1.0` by default. The end token is the final, true sentence period (not a decimal dot). Previous chosen response NLL already contains this period with a smaller weight; the added term explicitly increases its importance. Both positive and negative responses remain unchanged.

This is **not** the original instructor course loss, and **not** reference-model DPO. It is an optional ablation. `dpo/dpo.ipynb` remains unmodified (default `loss_mode='course'`).

## Run on TC1

```bash
cd ~/NanoGPT-Math
source .venv/bin/activate
git status --short
git pull --ff-only origin main

# CPU-only: exact dataset and hash validation
python scripts/overfit256_termination_aware.py --check-data-only

# One GPU; do not train from the TC1 login node
sbatch scripts/overfit256_termination_aware.sbatch
```

Check the Slurm job ID from `sbatch`, and then inspect:

```bash
cat term256_<JOB_ID>.out
tail -n 50 term256_<JOB_ID>.err
cat results/overfit256_termination_aware_summary.json
```

Outputs will be **isolated** in:

- `dpo/experiments/overfit256_termination_aware/dpo.pt`
- `dpo/experiments/overfit256_termination_aware/{train,val}_predictions.json`
- `results/overfit256_termination_aware_summary.json`

The runner refuses to overwrite previous experiment directories or existing summary files, and it does not edit source datasets, `dpo/dpo.pt`, or the course notebook.

## Key measures

1. **Teacher-forced P(.) after the GOLD answer** (and rate that the period is Top-1 there) — was termination learned at the desired position?
2. **Natural period-stop rate** — does generation now terminate, or does it keep repeating digits? High stop rate alone is insufficient: premature stops can also be wrong.
3. **Gold exact-number prefix frequency** — does the correct numeric string emerge before extra characters? This is not accuracy.
4. **Natural full-string exact answer accuracy** on **seen 256** and **held-out 256** — the primary outcome.
5. **Pairwise win rate** — did stronger ending supervision damage preference learning?

Interpretation:

| Result | Implication |
| --- | --- |
| P(.) at gold end rises; natural stops rise; exact accuracy rises | Partial success: better termination translates into correct generation |
| P(.) at gold end rises; natural stops rise; exact accuracy remains ~0 | Termination improved, mathematical digit sequence remains the bottleneck |
| Natural stops rise but many answers stop too early | Do not increase `gamma` blindly; analyze stop-position timing and numeral errors |
| P(.) stays low | Review mask alignment and optimization dynamics before scaling |

Future work, only after this check: diagnose digit position/carries, improve computational learning signal, then extend data to 20k/100k. Do not use held-out validation examples for retraining or hard-negative mining.

## Submission note

The Lab's required runnable notebook and Step 8 still need finalization. This standalone research diagnostic is *supplementary* to the teacher's loss, not a replacement for the required Notebook submission.
