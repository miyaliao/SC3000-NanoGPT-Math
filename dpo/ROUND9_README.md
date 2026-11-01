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

First mine a limited unique-question pilot: up to 50 questions per category (600 max), not 100k repeated rows:

    squeue -u "$USER"
    sbatch scripts/round9_mine.sbatch pilot

Inspect dpo/experiments/round9_mining/pilot/manifest.json and errors.json. Check how many REAL math-error completions are valid and how many are merely format failures. Full mode runs all eligible unique training prompts once and may be expensive; its runtime must fit TC1 QoS. Only proceed if pilot yield and examples justify it:

    sbatch scripts/round9_mine.sbatch full
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
