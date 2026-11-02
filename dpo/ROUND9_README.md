# Round9 — Structured Sampling × Frozen-Model Hard Negatives

Status: code on branch round9-balanced-self-hardneg. NO Round9 GPU training, data generation or new accuracy is claimed yet.

## Baselines
Round7 Replay: Val768 567, Old OOD256 109, Extended384 174 (Direct128 28).
Teacher680: Round7 Replay 498/680 (Core400 301, Extension280 197), versus Round5-A 481/680. Paired Round7-only 20; Round5-only 3. All 680 facts were previously seen, so this is competency/retention, NOT unseen-fact generalization.

Weak Teacher680 categories: a_minus_x 37/80, subtract 20/40, a_times_x 25/40, x_over_a 25/40.

## Four arms, all from frozen Round7 Replay
R (replay): 70k common older TRAIN + 30k other older TRAIN; refusal negatives.
U (random): 70k SAME common TRAIN + 30k newly drawn TRAIN-partition arithmetic questions; refusal negatives.
S (structured): 70k SAME common TRAIN + 30k from SAME eligible candidate pool, split by 12 category x 4 mathematical strata (625 each); refusal negatives.
H (hardneg): EXACT SAME 100k Prompts, Chosen and row order as S; replace only vetted model-wrong negatives for <=5000 presentations, <=5 per unique question, <=500 per category.

All use teacher original Course Loss, seed42, batch64, LR1e-5, one epoch, 1562 full updates, same strict answer-first evaluator and greedy period/newline stop.

The new candidates derive from sum_relations and product_relations, and use original canonical stable_split(seed42)==train. All new candidates exclude old Val768/OOD256/Extended384, course RESERVED, public Blind61/41, Teacher Core400 and Extension280. The historical shared 70k may contain previously Seen Teacher680 facts; do not relabel them as unseen. Round3 FINAL TEST768 is never read.

Additive strata: underlying max operand <=30/>30 x carries 0/>0 (inverse subtraction borrow). Multiplicative strata: product <=99, product 100-400 with 1-digit factor, product 100-400 with 2-digit factors, product 401-999. Dataset generator reports actual source and unique-fact/unique-prompt counts; 100k rows are NOT 100k unique problems.

## TC1 instructions

Activate existing venv, then switch to this experimental branch. Do not discard outstanding local changes:

    cd ~/NanoGPT-Math
    git fetch origin
    git switch --track origin/round9-balanced-self-hardneg
    source .venv/bin/activate

    python scripts/evaluate_teacher_style_680.py --check-only
    python scripts/build_round9_data.py --self-test-only
    python scripts/build_round9_data.py --verify-only
    python scripts/build_round9_data.py --build

    python scripts/train_round9.py --arm replay --device cpu --check-only
    python scripts/train_round9.py --arm random --device cpu --check-only
    python scripts/train_round9.py --arm structured --device cpu --check-only

Inspect dpo/data/round9_balanced_full/manifest.json and sample examples before GPU.

First mine a limited unique-question pilot: up to 50 questions per category (600 max), not 100k repeated rows. Observed on TC1 2026-11-01: 434/600 correct, 166/600 (27.67%) valid strictly formatted numerical errors. Error rates include add 4/50, subtract 11/50, multiply 30/50, x_over_a 23/50, x_minus_a 3/50. These are pilot sample rates, not guaranteed full-corpus rates:

    squeue -u "$USER"
    sbatch scripts/round9_mine.sbatch pilot

Inspect dpo/experiments/round9_mining/pilot/manifest.json and errors.json. Check how many REAL math-error completions are valid and how many are merely format failures. Full mode runs all eligible unique training prompts once and may be expensive; its runtime must fit TC1 QoS. Only proceed if pilot yield and examples justify it:

    # Full mining is split into restartable batches of up to 8,000 unique prompts.
    # Each job requests TC1's user-confirmed 6-hour maximum and checkpoints every 200.
    sbatch scripts/round9_mine.sbatch full
    # Once the previous job exits, if output ends in PARTIAL, submit again:
    sbatch scripts/round9_mine.sbatch full --resume
    # Repeat --resume (with at most one mining job active) until MINE PASS.
    python scripts/mine_round9_hardneg.py --assemble
    python scripts/train_round9.py --arm hardneg --device cpu --check-only

Run one small GPU smoke first (using an active Slurm GPU allocation):

    python scripts/train_round9.py --arm structured --smoke --device cuda

Submit at most two jobs at once under current TC1 limits (one GPU active):

    sbatch scripts/round9_train.sbatch replay
    sbatch scripts/round9_train.sbatch random

After completed jobs free queue slots:

    sbatch scripts/round9_train.sbatch structured
    sbatch scripts/round9_train.sbatch hardneg

Finally:

    python scripts/compare_round9.py --self-test-only
    python scripts/compare_round9.py --require-all

Compare R vs U (new data), U vs S (structured data), and S vs H (real wrong negatives), on frozen Val768 / OOD256 / Extended384 / Direct128 / Teacher680. Report paired only wins and regression counts. Keep Round3 final Test unused until final selection. Teacher680 measures Seen fact competency, not generalization. The TA's private grader is not verified; this is a uniform historical proxy.

## Caveats
A long model-generated Rejected versus a short refusal changes token weighting under instructor full-sequence mean loss, so S vs H measures a real-world replacement strategy, not a pure semantic hardness mechanism. HardNeg should NOT be invented when the frozen model answers correctly; the miner only accepts full, period-terminated, math-wrong answers. If fewer than 20 independent usable wrong prompts exist, stop instead of fabricating negatives.

## Files
scripts/build_round9_data.py — protected R/U/S generator.
scripts/mine_round9_hardneg.py — unique train prompt pilot/full mining + H construction.
scripts/train_round9.py — matched training + all fixed evaluations.
scripts/compare_round9.py — read-only paired comparison.
scripts/round9_mine.sbatch and scripts/round9_train.sbatch — TC1 runners.

## Random E2 restarted continuation (Round9 follow-up)

**Chosen checkpoint:** Round9 Random E1 (`dpo/experiments/round9_random/dpo.pt`) because historical strict Val768=578 and Extended384=176 are highest in Round9. Round7 Replay keeps a higher Teacher680=498 vs Random=489, so no broad-best claim. Existing R/U/S/H accuracy after one Epoch: R Val575 OOD110 Extended171 Teacher494; U 578/110/176/489; S 572/110/170/493; H 536/100/158/464. H is not selected.

**What exactly changes:** Start from the frozen Random E1 model weights and run the same 100k Random preference rows for one additional epoch. Keep same Course Loss, batch64, LR1e-5, seed42, 1562 updates, greedy stop and fixed Val768/OOD256/Extended384/Teacher680. Since Random E1 checkpoint did **not** save AdamW/scheduler state, this is an **optimizer/scheduler restarted continuation**, **not** a mathematically exact uninterrupted Epoch2; report the distinction. The script checks E1 manifest SHA, dataset SHA, original initialization SHA and parent checkpoint presence, and never overwrites `round9_random/`.

**TC1 (no other GPU job active; safely preserve local edits):**

    cd ~/NanoGPT-Math
    git status --short
    git pull --ff-only origin round9-balanced-self-hardneg
    source .venv/bin/activate

    python scripts/train_round9_random_e2.py --check-only

    # Small isolated GPU smoke is recommended before the full job:
    # inside a one-GPU Slurm allocation run:
    python scripts/train_round9_random_e2.py --smoke --device cuda

    sbatch scripts/round9_random_e2.sbatch

    # After the full job finishes and reports ROUND9 CONTINUATION PASS:
    python scripts/compare_round9_random_e2.py --check-only
    python scripts/compare_round9_random_e2.py

Outputs under `dpo/experiments/round9_random_e2_restart/` include `dpo.pt`, `summary.json`, all frozen predictions, and `paired_vs_random_e1.json` (comparison script). Stop further training if new Val768 gain is small or fact-disjoint OOD/Extended regresses materially. Do not use the sealed Round3 final Test for choosing this checkpoint.

## Multi-epoch Random E3–E6: checkpoint EACH epoch, evaluate AFTER training

As of 2026-11-02, completed E2 baseline from E1 Restarted Continuation:
TrainProbe256 216, Val768 582, OOD256 112, Extended384 180,
Teacher680 493, Direct128 32. E2 at
`dpo/experiments/round9_random_e2_restart/dpo.pt` is immutable.

**Training plan:** 4 EXTRA epochs (E3, E4, E5, E6), 100k *same* Random
positive/negative pairs, batch64, lr=1e-5, original Course Loss, seed42.
Initialize from the frozen Random E2 weights, initialize a *new* AdamW and
CosineAnnealingLR once, then keep both continuously across E3–E6 (total
4×1562 updates, not a scheduler reset at each epoch). E2 did not save optimizer
state, therefore E2→E3 is a restarted continuation, but E3→E4→E5→E6
does preserve continuous AdamW/scheduler state inside this run.

Separate train/evaluate Slurm jobs respect TC1's 6-hour per-job QoS limit.
The TRAIN script performs no evaluation inference: saves
`epoch_e03.pt`, `epoch_e04.pt`, `epoch_e05.pt`, `epoch_e06.pt`
and `state_eXX.pt` (optimizer/scheduler/Python+Torch RNG/actual shuffled row
order to allow exact epoch-boundary resume). Output in
`dpo/experiments/round9_random_multi_e3_e6/`. An append-only
`train_progress.json` records each finished epoch and SHA256;
`train_manifest.json` appears only after *all* four epochs finish.
No old checkpoint or evaluation is overwritten.

Commands, from TC1 repository root on branch `round9-balanced-self-hardneg`:

    git status --short
    git pull --ff-only origin round9-balanced-self-hardneg
    source .venv/bin/activate
    python scripts/train_round9_random_multi.py --check-only
    # optional short smoke (separate directory; limited 128 rows, 2 epochs):
    sbatch scripts/round9_random_multi_train.sbatch --smoke
    # after smoke job has finished and no GPU job is active:
    sbatch scripts/round9_random_multi_train.sbatch
    # if a job is stopped after one or more complete epochs:
    sbatch scripts/round9_random_multi_train.sbatch --resume

When `MULTI TRAIN PASS` appears, run CPU verification and submit ONE
GPU evaluation job to **evaluate all four saved epoch checkpoints afterward**:

    python scripts/eval_round9_random_multi.py --check-only
    sbatch scripts/round9_random_multi_eval.sbatch

The evaluation job uses the same frozen TrainProbe256, Val768, OldOOD256,
Extended384 and Teacher680 as before, greedy/strict answer-first, 80 tokens.
Direct128 is an exact subgroup of Extended384, scored without re-inference.
Each epoch's predictions/summaries are saved immediately under
`round9_random_multi_e3_e6/eval/e03/` etc.; another invocation safely
skips completed epochs/splits. Full E2-vs-E3..E6 paired comparisons and
per-category scores are saved to
`round9_random_multi_e3_e6/eval/all_epochs_comparison.json`.
The sealed Round3 final Test is never accessed. **No accuracy claims until
training and evaluation have been run on TC1.**

Job monitoring (substitute the actual printed job ID):

    squeue -u "$USER"
    tail -n 30 -f round9_random_multi_train_<jobid>.out
    tail -n 30 -f round9_random_multi_eval_<jobid>.out

A longer run should not be justified solely by validation-set tuning:
avoid selecting epochs using the sealed Test, and report potential
validation selection bias if choosing the best among several snapshots.
