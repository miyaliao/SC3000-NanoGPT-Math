# Round 7 — Failure-Pattern Targeted Fine-Tuning (proposal / diagnostic phase)

**Status (2026-10-30):** Round5-A error-type audit completed by user; Round7 TRAIN-only data generator, matched trainer, Slurm jobs and paired comparison scripts are committed. **Neither the new Round7 dataset nor GPU checkpoint has been generated/run yet.** Do **not** train on validation failures or the public Blind61/Blind41 examples. The user must run the preflight and training commands described below.

## Why a new round is reasonable

Our strongest conventional output-format checkpoint, **Round5-A**, scored:
- Round3 Val 560/768 = 72.92%
- Old OOD 106/256 = 41.41%
- Extended multiplication 168/384 = 43.75%
- Prior-year public Blind61/Blind41, greedy 50/61 and 29/41

In the public test, Round5-A greedy correct on previously seen mathematical facts was 44/49 and 23/26, while unseen math facts were 6/12 and 6/15. These small, curated subgroups suggest a generalization gap; training on their individual prompts would contaminate the public comparison.

Past Round3/4 audits also found disproportionately poor results for larger products and reverse-multiplication, plus difficult carry/borrow cases. **The exact current Round5-A error-distribution needs to be measured before choosing new quotas**; do not assume every three-digit result is difficult.

## Step 1: CPU diagnosis from existing saved predictions

```bash
cd ~/NanoGPT-Math
git pull --ff-only origin main
source .venv/bin/activate
python scripts/audit_round7_failure_patterns.py
python scripts/audit_round7_failure_patterns.py --save
```

Read-only sources include the existing **Round5-A** Val768, Old OOD256, ExtendedVal384 and public Blind61/Blind41 greedy predictions. The script reports category, output format vs numeric math error, multiplicative target (large product vs inverse factor), product magnitude, factor type, carry/no-carry, and number length. It does not read the sealed final Test or load model weights. Public-question error analysis is descriptive only, and may NOT be used to select individual new training examples or maximize public scores.

Output: `results/round7_failure_patterns.json`, saved only if absent.

## Step 2: Analysis-driven choice (fixed from actual audit)

Round5-A performance by skill:
- Val768: add 55/64 (85.9%), multiply 37/64 (57.8%), `x_over_a` 36/64 (56.2%), direct product 73/128 (57.0%), inverse factor 178/256 (69.5%).
- Val math family: product <=100 correct138/145 (95.2%), product101–400 correct113/239 (47.3%).
- Old OOD: product >=1000 correct0/31; direct product8/48 (16.7%).
- Extended384: multiply16/64 (25.0%), `x_over_a`10/64 (15.6%), direct product26/128 (20.3%), inverse factor142/256 (55.5%).
- Zero format failures: failures are **numerical**, not response-format failures. The crude carry/no-carry tag did NOT show a clear additive carry bottleneck; do not divert the next experiment toward general addition/borrow exercises.

### Round7 fixed, matched two-arm data design

Same **Round5-A** starting model, one Course Loss epoch, batch64, LR1e-5, seed42, same scheduler, same 100,000 total rows and 1,562 full optimizer steps:

| Dataset component | Replay-only CONTROL | Targeted TREATMENT |
|---|---:|---:|
| Shared original Round5-A training rows | 70,000 | **Same** 70,000 |
| Other ordinary original TRAIN rows | 30,000 | 0 |
| Existing hard original TRAIN-only multiplication rows (reweighted) | 0 | 15,000 |
| NEW TRAIN-partition two-digit-by-two-digit multiplication facts, products 1000–4999 | 0 | 15,000 |
| **Total** | **100,000** | **100,000** |

Both 15,000-row treatment components have the SAME category quotas:
- `multiply`: 3,750
- `x_over_a`: 3,750
- `x_times_a`: 2,250
- `a_times_x`: 2,250
- `divide`: 1,500
- `a_over_x`: 1,500

No new loss or answer template: Positive is `The answer is N because ...`, Negative remains instructor-compatible refusal. Every new mathematical fact is verified using `parse_relation`, `stable_split(key,42)=='train'`, and explicit exclusion of frozen Round3 Val768, OOD256 and Extended384. The sealed Round3 final Test and public Blind61/41 are **not read**. The original partition function makes the new data disjoint from Round3 independent final Test by construction. Source SHA checks, tokenizer/context constraints, per-category pool capacity and unique math fact counts are audited before writing.

**The intervention is compound** (novel product-range exposure + hard-example reweighting), so this experiment measures the bundle against replay, not the individual causal contributions of either component.

## Step 3: CPU verification and generating training-only data

```bash
cd ~/NanoGPT-Math
git pull --ff-only origin main
source .venv/bin/activate

python scripts/build_round7_targeted_data.py --profile mini --verify-only
python scripts/build_round7_targeted_data.py --profile full --verify-only
python scripts/build_round7_targeted_data.py --profile full

python scripts/train_round7_matched.py --arm replay --check-only
python scripts/train_round7_matched.py --arm targeted --check-only
```

All commands above require no GPU and must pass before submitting full jobs. Paths are `dpo/data/round7_targeted_full/{replay_pairs.json,targeted_pairs.json,manifest.json}`; all generated data are gitignored and their SHA digests recorded. Generation REFUSES to overwrite an existing folder. Neither the original Round5-A checkpoint nor any previous training data or predictions are changed.

## Step 4: Run TWO matched Slurm jobs (TC1 maximum two queued, one running)

If the CPU preflight is clean and no existing jobs occupy the 2-job QoS quota:

```bash
squeue -u "$USER"
sbatch scripts/round7_matched.sbatch replay
sbatch scripts/round7_matched.sbatch targeted
```

One job per arm, identical `--epochs 1 --batch-size 64 --lr 1e-5 --max-new-tokens 80`. Each writes to a NEW output folder `dpo/experiments/round7_{replay,targeted}/`, including model checkpoint, paired-strict evaluation on frozen Val768/OOD256/Extended384/Probe256, saved predictions and summary. Full training only on TC1 GPU; do not start on login node. Never overwrite an existing output folder or rerun a successful arm by accident.

After both finish:

```bash
python scripts/compare_round7_matched.py
```

Compare **TARGET minus REPLAY** (primary causal comparison), not TARGET minus Round5-A alone. Also report both relative to the frozen Round5-A baseline Val560/768, OOD106/256 and Extended168/384. Crucial checkpoints: direct large-product128 within Extended384, and whether in-domain improvements are offset by OOD/generalization losses. Use the SAME strict answer-at-start parser for both arms. The previous public Blind61/41 outputs and sealed Round3 final Test are not used for Round7 model selection.

## Avoid the tempting invalid design

Do **not** take any failed question from `val_predictions.json`, `old_ood_predictions.json`, `extended_val_predictions.json` or `round5_A_greedy_blind*_predictions.json` and insert the exact question or its inverse/commuted mathematical relation into the next training data. That would artificially improve the metric we use to claim generalization.

The 2025 prior-year repository focused on dense deterministic arithmetic patterns and trained more total epochs, but its sampled curated 41-item blind set included at least 20 exact prompts guaranteed by its own deterministic training generator. We should borrow the **teaching curriculum idea** without introducing the same train-test contamination.

**Next action:** run `scripts/audit_round7_failure_patterns.py` on CPU and paste its category/feature breakdown. Then finalize quotas and implement a paired fact-disjoint generator/trainer.
