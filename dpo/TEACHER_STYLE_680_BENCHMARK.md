# Teacher-style Benchmark v1: Core400 + Extension280 = 680 questions

**Design locked 2026-10-30.** The original [Core400](TEACHER_STYLE_400_BENCHMARK.md) is already frozen and tracked, with SHA256 `c27518670be350154e9eb8d93ec7126a518c4d3cd4cf23a942649970011e4396`. **DO NOT edit or regenerate it.** The extra seven SC3000 math families are held in a *separate* `Extension280` freeze and must be generated and committed on TC1 before model inference or Round8 training.

The stratification is **inspired by** the digit/carry analysis in [Teaching Arithmetic to Small Transformers, ICLR 2024](https://github.com/lee-ny/teaching_arithmetic). It is **not the paper's raw benchmark**, whose arithmetic prompt/answer formats and task scope differ from SC3000.

## Exactly 280 new prompts: 7 original course formats × 40 each

Each operation has **four subskills**, with **10 distinct prompt strings and 10 distinct math facts within that subskill**:

| Operation | Subskills (10 each) | Examples |
|---|---|---|
| `subtract` | Result 1/2 digits × ones-column borrow yes/no | `52-19=?` |
| `x_plus_a` | Unknown x is 1/2 digits × units-column carry yes/no | `x+17=36,x=?` |
| `a_plus_x` | Unknown x is 1/2 digits × units-column carry yes/no | `17+x=36,x=?` |
| `x_minus_a` | Subtrahend a is 1/2 digits × units-column borrow yes/no (x itself 2 digits) | `x-17=20,x=?` |
| `a_times_x` | x 1 digit and known factor 2 digits; x 2 digits and known factor 1 digit; both factors 2 digits with product 101–400; both factors 2 digits with product 401–999 | `11*x=44,x=?` |
| `x_over_a` | divisor 1 digit / quotient 2 digits; divisor 2 digits / quotient 1 digit; both factors 2 digits with product 101–400; both factors 2 digits with product 401–999 | `x/4=18,x=?` |
| `a_over_x` | unknown divisor 1 digit / quotient 2 digits; unknown divisor 2 digits / quotient 1 digit; both factors 2 digits with product 101–400; both factors 2 digits with product 401–999 | `72/x=4,x=?` |

No negative or fractional arithmetic; all multiplication-derived correct results **≤999**, no four-digit products. For the small equation classes, operands are mainly 1–2 digits. If a fact can be phrased in multiple inverse math forms, its **canonical key** is used to enforce leakage exclusions.

The initial idea of separating `1digit × 1digit` multiplication within these *new* classes was rejected because the frozen Core400 already protects almost the entire finite easy multiplication-table fact space. The final four classes instead retain elementary 1×2 / 2×1 fact types and split 2×2 multiplication by the result range; this preserves non-overlap from Core400 and avoids artificial four-digit problems.

## Isolation, status labels, and reproducibility

1. Every candidate has independently verified exact integer gold with `verify_math` and `parse_relation`.
2. Exclude **ALL canonical math facts already represented in Core400**, not only its prompt strings.
3. Also exclude math facts in historic Round3 Val768, Old OOD256, Extended Val384, published prior-year blind tests, and instructor sample prompts.
4. Every new selected fact satisfies `stable_split(fact,42)=='train'`, which avoids the original sealed Test partition **without reading the sealed Test file**.
5. Label Seen versus Unseen according to the **actual historical Round3/Round4/Round5-A training union**. Core400 was **Seen400/400**. The extension may also have few/zero Unseen because it intentionally focuses on easy arithmetic. Never call a Seen score a test of unseen-fact generalization.
6. All **280 prompt strings are distinct**, and the two benchmark parts together have **680 distinct prompt strings and NO shared canonical math facts**. A single canonical math fact may appear as two *different extension prompts* across equation forms; this is disclosed in Extension280's manifest, not claimed to be 680 independent facts.
7. The generator uses deterministic seed `20261031`, has four equal strata of 10 per category, saves a per-file SHA256 manifest, refuses overwrite, and reports actual Seen/Unseen numbers. It fails if any stratum cannot be filled legally; **do not remove leakage protections or replace subskills after seeing scores.**
8. Both frozen parts are **development/competency** benchmarks selected with knowledge of broader past difficulties; they are not the TA hidden test. Keep the independent sealed Round3 Test unopened.

## Step 1: CPU-only preflight and freeze Extension280 ONCE

```bash
cd ~/NanoGPT-Math
git pull --ff-only origin main
source .venv/bin/activate

python scripts/build_teacher_style_extension_280.py --self-test-only
python scripts/build_teacher_style_extension_280.py --audit-only
```

The audit must show **AUDIT PASS** with 7 groups ×40 and 28 subskills ×10. Then freeze and verify:

```bash
python scripts/build_teacher_style_extension_280.py --freeze
python scripts/build_teacher_style_extension_280.py --check-frozen
python scripts/evaluate_teacher_style_680.py --check-only
```

The new files are:

```text
dpo/benchmarks/teacher_style_extension_280_v1/questions.json
dpo/benchmarks/teacher_style_extension_280_v1/manifest.json
```

Save the exact frozen dataset to GitHub (these files are **NOT gitignored**):

```bash
git add dpo/benchmarks/teacher_style_extension_280_v1/questions.json \
        dpo/benchmarks/teacher_style_extension_280_v1/manifest.json
git commit -m "benchmark: freeze 280 extension problems, complete teacher-style 680"
git push origin main
```

**Never rerun `--freeze` when the directory exists or tune the 280 questions after seeing model scores.** The original Core400 remains unchanged.

## Step 2: one-GPU matched score across BOTH frozen files

```bash
squeue -u "$USER"
sbatch scripts/teacher_style_680_eval.sbatch
```

This uses a **single Slurm GPU job** to evaluate Round5-A and Round7 Replay on the exact same 680 questions, with greedy decoding (original NanoGPT first-period/newline termination), `max_new_tokens=80` and uniform anchored answer-first primary scorer. This does **not** verify the TA's undisclosed extraction.

Read:

```bash
grep -E "FINAL |PAIRED |SAVED" teacher680_*.out
cat dpo/experiments/teacher_style_680_v1_eval/summary.json
```

Output includes 680 overall, Core400, Extension280, all 12 math categories, all 48 subskills (20 Core+28 extension), Seen vs Unseen, and **paired-only correct differences between two checkpoints**. It saves all 680 individual questions, outputs, correctness statuses and gold answers. Inference outputs are gitignored; do not commit model weights.

The old Core400 standalone evaluator is preserved and not changed. Its scores should agree with the Core400 portion of the new 680 job when using the same model and decoding configuration.

## Step 3: protect new benchmark from Round8 self-error mining

`scripts/mine_round8_train_hard_negatives.py` now requires the **two frozen parts** with intact SHA256 before starting any GPU mining. It excludes the canonical math facts represented by **all 680 prompts** from selected TRAIN errors and their number variants. The subsequent paired training builder revalidates this protection before training. Do NOT run old mining jobs before freezing Extension280.

Project decision: extend an existing immutable benchmark; do not erase or replace the original 400, and do not claim this historical Seen-heavy competency suite is a strict unseen test.
