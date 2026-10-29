# SC3000 Overfit-256 termination gamma sweep (0.25 / 0.50 / 1.00)

This is a **one-variable ablation** of the additional gold period-loss coefficient.
The gamma=1.0 run already completed at 16/256 training accuracy, 0/256 validation,
256/256 period stops, 16/256 gold number prefixes, P(period after gold)=0.9735,
and 252/256 preference wins.

## What is held fixed

All runs use exactly the same 256 Round 1C preference pairs and 256 frozen validation
questions, seed 42, original sft/gpt.pt initialization (not continued fine-tuning),
128 epochs, batch 64, 512 optimizer steps, beta=0.5, answer_weight=1.0,
completion_weight=0.25, optimizer/scheduler/model, and original-period-stop
greedy evaluation.

Only end_weight (gamma) is changed: 0.25 or 0.50 vs completed 1.00.
The experiment is supplemental and is *not* the instructor's original course Loss.
The main dpo/dpo.ipynb remains unchanged.

## TC1 launch (one GPU maximum per user)

```bash
cd ~/NanoGPT-Math
git status --short
git pull --ff-only origin main
source .venv/bin/activate
python scripts/overfit256_termination_aware.py --check-data-only

sbatch scripts/overfit256_termination_gamma050.sbatch
sbatch scripts/overfit256_termination_gamma025.sbatch
squeue -u "$USER"
```

You may submit both jobs at the same time, but under TC1's one-GPU-per-user
policy they will normally execute **sequentially**. Both reserve only one GPU.
Do not launch CUDA training on the login node.

## Independent artifacts

| gamma | checkpoint | summary | log |
|---|---|---|---|
| 0.50 | dpo/experiments/overfit256_termination_gamma050/dpo.pt | results/overfit256_termination_gamma050_summary.json | term_050_JOBID.out |
| 0.25 | dpo/experiments/overfit256_termination_gamma025/dpo.pt | results/overfit256_termination_gamma025_summary.json | term_025_JOBID.out |
| 1.00 | dpo/experiments/overfit256_termination_aware/dpo.pt | results/overfit256_termination_aware_summary.json | term256_JOBID.out |

The runner refuses to overwrite either an existing checkpoint directory or summary.
All three outputs include train and validation predictions plus preference and end statistics.

After the two jobs complete:

```bash
python scripts/compare_termination_gamma.py
```

Pay special attention to train exact accuracy, gold-answer prefix rate, natural
period stops, end probability, and validation accuracy. A higher period stop
rate by itself is not an improvement in mathematics. The original TA grading
script remains undisclosed: this uses deterministic greedy with instructor
period/newline stop semantics.

If an output directory already exists because of a failed earlier attempt,
inspect its contents and logs rather than removing/overwriting it blindly.
