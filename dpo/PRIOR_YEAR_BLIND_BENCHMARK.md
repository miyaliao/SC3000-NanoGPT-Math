# SC3000 — Same-question comparison against published prior-year GitHub results

Public reference: [JamesRobertJohns/3000](https://github.com/JamesRobertJohns/3000), Jupyter notebook `src/dpo/dpo.ipynb`.

**Source-derived reference scores reverified using actual saved Notebook outputs:**
- Published `test_set_4`, 61 distinct blind questions, **52/61 = 85.25%**, tested on their model **before** extra fine-tuning.
- Published `blind_test_set`, 41 distinct blind questions, **39/41 = 95.12%**, tested on their model **after** extra fine-tuning.
- The two lists share **one identical prompt**, `33+66=?`, i.e., 101 distinct prompts across 102 entries.
- The reference sampled with `temperature=0.8`, `top_k=200`, `max_new_tokens=200` using `model.generate` and printed one output per prompt. Published outcomes are not a deterministic metric or an official TA hidden test.
- The original Notebook does NOT demonstrate fact-disjointness of these particular handpicked tests from all training records, and the published model changes between 61/41. Do not pool these as one IID study or claim 95.12% is a fixed independent benchmark.

## Frozen artifact

`dpo/reference_2025_blind_sets.json` contains every one of the 61+41 source questions in original order, independently computed mathematical gold answer, published completion and parsed published answer, and score. Source Notebook GitHub blob SHA is recorded in metadata. Do not edit frozen tests to improve scores.

## Our locally trained models to compare

| Model | Local TC1 checkpoint | Answer template |
|---|---|---|
| Round5-A (historical in-domain best) | `dpo/experiments/round5_original/dpo.pt` | one-step `because` |
| Round6-A (5 epochs from SFT) | `dpo/experiments/round6_from_sft/original/dpo.pt` | one-step `because` |
| Round6-D (5 epochs from SFT) | `dpo/experiments/round6_from_sft/decomposition_before/dpo.pt` | pre-answer decomposition on 16,271 training rows |

The large checkpoint weights are **not uploaded to GitHub**. The following model inference must be run on TC1 with access to the files.

## CPU audit before submitting GPU job

```bash
cd ~/NanoGPT-Math
git pull --ff-only origin main
source .venv/bin/activate
python scripts/benchmark_prior_year_blind.py --self-test-only
python scripts/benchmark_prior_year_blind.py --audit-only
```

The first command independently checks all 102 source gold answers, published completions and the 52/61,39/41 scores. The second scans **all our original Round4 Treatment 100k training records** used by all three models, and counts for each reference group:
- Exact prompt repeated in our training records
- Mathematical fact repeated in our training records, including `a*b=c` vs `c/a=b` and equation rearrangements (canonical fact key)
- Facts absent from our training (critical generalization subset)

**These are existing public tests, not our independent sealed Round3 final Test.** Results on seen facts cannot establish out-of-training fact generalization. The original reference repo's training overlap cannot automatically be inferred from ours.

## Matched GPU inference for three existing checkpoints

```bash
squeue -u "$USER"
sbatch scripts/prior_year_blind.sbatch
```

TC1 limits 1 running GPU and ~2 submitted jobs per user; submit only if there is a queue slot. This runs **both** modes for each checkpoint across all 61+41 examples:
- `greedy` = exact `model.py` period/newline stop + argmax; matches our Round3–6 deterministic comparison convention.
- `sampled` = actual `model.generate()` with reference `temperature=0.8, top_k=200`, max 200 generated characters, seeded per prompt. It matches reference sampling **settings**, not its unreported RNG state nor its actual model.
- All our three models receive identical prompts and the SAME legacy strict answer-first number scoring. D's `final_status` is retained **only as exploratory diagnostic** and is not mixed with the strict primary rankings.
- Both modes record per-question output and per-question training overlap. Print n correct and accuracy for each group, model and mode, and separately performance on held-out *mathematical facts*.
- All evaluation outputs go only to **new** `dpo/experiments/prior_year_blind_2025/`. Refuses to overwrite. No training is performed. The sealed internal `test.json` is not touched.

Output includes `summary.json`, `source_reference_and_leakage.json`, and per-model/per-mode/per-group `*_predictions.json`.

```bash
cat dpo/experiments/prior_year_blind_2025/summary.json
grep "PRIMARY=" prior_blind_*.out
```

For fewer GPU calculations, you may instead run only greedy:
```bash
python scripts/benchmark_prior_year_blind.py \
  --device cuda --decode greedy \
  --out-dir dpo/experiments/prior_year_blind_greedy_only
```
Run the above **inside a Slurm GPU allocation**, not the login node.

## Interpretation

A result greater than 52/61 or 39/41 on the same test prompts would exceed the previous **published sample accuracy for those prompts under that specific decoding protocol**. It would NOT prove globally superior math ability: different decoding randomness/checkpoints, manually selected problems, and mathematical-fact train overlap still matter. To choose our submission model use our existing disjoint Val768, Old OOD256 and Extended Val384, plus the TA's permitted answer format; keep sealed Test untouched until after selection.
