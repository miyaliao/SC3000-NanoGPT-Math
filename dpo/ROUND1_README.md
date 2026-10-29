# SC3000 NanoGPT-Math — Round 1 (refusal preference)

Round 1 uses the successful pipeline-smoked Notebook **Step 2–7 verbatim**.
It does not require extra SFT or a reference model.

## Generate and audit (CPU/login node)

```bash
cd ~/NanoGPT-Math
source .venv/bin/activate
python scripts/generate_round1.py --profile mini
python scripts/generate_round1.py --profile full --install-train
```

The full output goes to `dpo/data/round1_full/`. The final `--install-train` only
copies its verified training JSON to the path your Notebook Step 5 actually
reads: `dpo/pos_neg_pairs.json`. The first 4 course examples are backed up at
`dpo/data/course_examples.json` if they were still there. The generated data
are deterministic (seed 42). Re-running does not need an internet connection.

Expected counts: train 20,480; validation 2,048; held-out test 2,048.
Train has 85% integer and 15% finite decimal cases, split equally between
arithmetic and one-variable equation types. Positive: `The answer is X.`;
negative: `Sorry, I don't know.`. Mathematical facts, not just prompt strings,
are isolated across data splits.

## First evaluate the unchanged pretrained baseline (one GPU)

Use a stable validation subset, never train on it:

```bash
python scripts/evaluate_math.py --test dpo/data/round1_full/round1_val.json \
  --checkpoint sft/gpt.pt --out results/round1_val_baseline_256.jsonl \
  --limit 256 --device cuda
```

For the **final** test set, evaluate only after fixing the model selection
procedure. Do not mine negatives from validation or test questions.

## Train Round 1 (one GPU Slurm job)

```bash
sbatch scripts/round1_train.sbatch
squeue -u "$USER"
tail -n 30 round1_<JOB_ID>.out
```

This executes the actual Notebook Step 2–7 cells with 5 epochs, batch size 64,
320 batches per epoch, and the existing teacher-recommended preference loss.
Outputs: `dpo/dpo.pt` and `results/round1_training_summary.json`. The runner
refuses to overwrite an existing formal checkpoint without `--overwrite`.

## Evaluate Round 1 using exactly the same validation questions

```bash
python scripts/evaluate_math.py --test dpo/data/round1_full/round1_val.json \
  --checkpoint dpo/dpo.pt --out results/round1_val_dpo_256.jsonl \
  --limit 256 --device cuda
```

Comparing loss is not enough. Report accuracy, refusal, mathematical error and
format error; compare on **the exact same 256 evaluation problems**.
The current `model.py` still stops at the first period. The evaluator's own
`math_eval_utils.greedy_completion()` peeks past numeric periods and grades
finite decimals exactly. This does not change the course model or loss.

## Submission/reproducibility

Include both the generator source and generated `pos_neg_pairs.json`, plus the
finished Notebook and `dpo/dpo.pt` in the final submission ZIP. Git tracks
scripts and data but `dpo/dpo.pt` is intentionally ignored; don't accidentally
commit a 100+ MiB checkpoint. Keep generated `round1_manifest.json` as proof
of dataset statistics and hashes. The test data is for final evaluation only.

Important: Notebook Step 8 is still a TODO and must be completed before final
submission. `train_round1.py` runs Steps 2–7 only and is an execution helper;
it does not replace the course's required executable Notebook.
