# Teacher-style 400 v1 — FROZEN evaluation protocol

**Design approved 2026-10-30. Actual 400 questions need to be materialized on TC1, because the 100k original train rows and protected validation data are only local to TC1.** After successful generation, commit `questions.json` and `manifest.json` and never regenerate them or tune their composition based on model performance.

Inspired by Nayoung Lee et al., **Teaching Arithmetic to Small Transformers** (ICLR 2024):
- [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6bf82fdcbd92b6a7793b3894422d2437-Abstract-Conference.html)
- [Official code](https://github.com/lee-ny/teaching_arithmetic), notably `data/create_multidigit_examples.py` and `data/bal/creating_balanced_data.ipynb`.
- The paper's actual plain-text arithmetic formats (e.g. `17+19=36` or reverse-order answers) are **NOT** direct drop-in SC3000 tests; this benchmark reuses the **per-digit/per-carry stratification concept**, with exactly the instructor's five example prompt structures.

## FIXED v1 specification: 5 operations × 80 = 400 questions

| Operation (SC3000 original prompt syntax) | Formula/range | Four subskills (20 rows each) |
|---|---|---|
| `add` | `a+b=?`, a,b two-digit 10..99, output 20..198 | no/yes units carry × no/yes tens carry |
| `multiply` | `a*b=?`, a 2..9, b 10..99, 2–3-digit product | 2/3 answer digits × units multiplication carry yes/no |
| `divide` | `n/d=?`, 10≤n≤99, 2≤d≤9, n divisible by d | one/two-digit quotient × even/odd divisor |
| `a_minus_x` | `a-x=b,x=?`, 20≤a≤99, 10≤b<a, x=a-b | 1/2 answer digits × ones-place borrowing yes/no |
| `x_times_a` | `x*a=b,x=?`, 2≤a≤15, 1≤x<80, a*x≤999 | 1/2 answer digits × 1/2 multiplier digits |

The five *distinct* instructor examples (`17+19=?`, `3*17=?`, `72/4=?`, `72-x=34,x=?`, `x*11=44,x=?`) are deliberately excluded from the 400 rows and are kept only as a **separate 5-question sanity check**. The repeated instructor examples do not inflate sample size.

A carry is counted using the published identity

`count_carries(a,b) = [digit_sum(a) + digit_sum(b) - digit_sum(a+b)] // 9`.

This is DIRECTLY based on the paper's addition data code, whereas our division/equation strata are SC3000-specific adaptations, NOT claims about an official ICLR dataset.

### Seen and Unseen math facts, transparently reported

Each of the four subskills contains 20 **unique canonical math facts**, and the benchmark contains exactly **400 unique math facts** across categories and commuted/inverse prompts. Select as close as possible to 10 Seen and 10 Unseen PER SUBSKILL **when feasible**, otherwise fill the fixed 20 from available source candidates and report the exact counts. "Seen" means the mathematical fact appeared in ANY of the actual Round3, Round4 Treatment or Round5-A original training sources used to reach Round5-A/Round7 Replay; "Unseen" means not in that historical union. Labels are not guessed from surface prompt spelling.

**Isolation protections:**
1. No mathematical fact is shared with frozen Round3 Val768, Old OOD256, Extended384, saved public prior-year Blind61/41 or instructor RESERVED samples.
2. All 400 mathematical facts must satisfy the original `stable_split(fact_key,42)=='train'` partition. This keeps them away from the **original sealed Round3 independent TEST partition WITHOUT OPENING its contents**; even novel unseen questions are generated only from *train-partition potential facts*, then withheld from any new train.
3. Once frozen, ALL 400 canonical facts (even historically Seen ones) are considered **protected** and future Round8 mining / numerical variants may not train on them. `scripts/mine_round8_train_hard_negatives.py` now REQUIRES the frozen benchmark to exist and verifies its SHA before mining.
4. The benchmark is a **development evaluation suite**, since it was designed using previous skill-level error analysis. It is not an independent hidden grader, and the sealed Round3 final Test remains untouched.
5. The historical `Seen` subgroup is **not a generalization test**, even though the benchmark is now frozen against future training. Report Seen and Unseen accuracy separately, plus every category and 20-row subskill.
6. If any bucket has fewer than 20 eligible distinct facts after all protections, the generator FAILS rather than silently changing bucket sizes or leaking other splits; inspect the printed shortage and revise the written specification deliberately **before** committing the frozen set.

## CPU: create benchmark exactly ONCE

```bash
cd ~/NanoGPT-Math
git pull --ff-only origin main
source .venv/bin/activate
python scripts/build_teacher_style_400.py --self-test-only
python scripts/build_teacher_style_400.py --audit-only
python scripts/build_teacher_style_400.py --freeze
python scripts/build_teacher_style_400.py --check-frozen
```

All these commands are **CPU-only**. Freezing writes:
- `dpo/benchmarks/teacher_style_400_v1/questions.json` — 400 labeled prompt/gold/category/subskill/fact-key/Seen-Unseen records;
- `dpo/benchmarks/teacher_style_400_v1/manifest.json` — frozen stratification counts, SHA256 of all source data, question-file SHA256, split/protection rules, seed and paper attribution.

The generator refuses to overwrite existing output. Check the counts in both console output and manifest. If they look right, immediately commit **the two small frozen files**:

```bash
git add dpo/benchmarks/teacher_style_400_v1/questions.json \
        dpo/benchmarks/teacher_style_400_v1/manifest.json
git commit -m "benchmark: freeze 400 teacher-style arithmetic questions, math fact isolated"
git push origin main
```

Once committed, **NEVER re-sample because a checkpoint scored poorly.** New rounds must not include those 400 math facts in any training Chosen/Rejected exercises, or nearby inverse forms.

## Later: matched R5-A and R7 Replay inference on TC1

Only after benchmark freeze, with one free GPU and no duplicated job:

```bash
squeue -u "$USER"
sbatch scripts/teacher_style_400_eval.sbatch
```

Read:
```bash
cat dpo/experiments/teacher_style_400_v1_eval/summary.json
```

The script uses exactly the previous model.py **greedy argmax, stop on '.'/newline**, and the SAME strict answer-first numeric score on BOTH Round5-A and Round7 Replay, at max_new_tokens=80. It prints 400 total accuracy, Seen/Unseen, operation 80, and every digit/carry subskill20; it saves per-question predictions in a new experiment-only folder. It NEVER reads or tunes against the sealed independent Test or public prior-year test prompts.

**Important:** the TA's actual hidden grading protocol is unknown; no benchmark score implies an official grade. We are evaluating our own fixed, instructor-style, scientifically stratified **development set**.
