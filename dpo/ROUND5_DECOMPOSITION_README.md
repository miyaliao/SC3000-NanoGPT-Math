# Round 5 — Explanation Decomposition A/B (pilot)

## Research question

Can exact, distributive **arithmetic decomposition after the answer** improve NanoGPT's ability to predict larger products, compared with the one-step `because a*b equals c` explanation? This is an ablation of **explanation content**, not a new Loss or a new set of math questions.

Round4 Treatment E1 is the fixed starting point. Its 384-question extended validation score was 172/384 = 44.79%; direct product-answer categories were multiply 15/64 and x_over_a 12/64. The other four classes were easier because their numeric answer was a smaller factor.

## What is in the 100k training data?

Exactly the Round4 Treatment training rows **in their original order**:

| Source | Rows |
|---|---:|
| Shared Round3 training examples | 70,000 |
| Harder previous multiplication/inverse examples | 15,000 |
| New product 401–999 and inverse-equation examples | 15,000 |
| Total (each A/B arm) | 100,000 |

Both arms use the same rows/negatives/correct answers. **Only B transforms eligible Positive explanatory tails**. Exact transformed row count is printed after generating from TC1's source dataset; do not infer this from category quotas because many old examples already match, and some are excluded.

**Eligible for decomposition:** only direct large-product-answer types `multiply` (a*b=?) and `x_over_a` (x/a=b,x=?); correct product >=100, each factor >=2, and at least one operand has a nonzero ones part that permits a nontrivial tens+ones split. Other 10 question categories, trivial identities, 1-digit products, and factors not meaningfully splittable stay unchanged.

Example:

A: `23*17=? The answer is 391 because 23*17 equals 391.`

B: `23*17=? The answer is 391 because 23*10=230 and 23*7=161 and 230+161=391.`

Also transforms `x/17=23,x=?` with correct answer 391; the other equation variants are retained without a different explanation. All partial computations and results must match the original gold answer exactly.

Note: `because` follows the first answer in autoregressive generation. It cannot serve as a scratchpad *before* the answer; this tests whether the auxiliary **training signal** improves prediction. Long explanations also change whole-sequence mean Course Loss weighting. Therefore even a positive result would not prove the model learned an inference-time reasoning process.

## CPU phase on TC1

```bash
cd ~/NanoGPT-Math
git status --short
git pull --ff-only origin main
source .venv/bin/activate

python scripts/test_round5_decomposition.py
python scripts/build_round5_decomposition.py --verify-only
python scripts/build_round5_decomposition.py

python scripts/train_round5_decomposition.py --arm original --check-only
python scripts/train_round5_decomposition.py --arm decomposition --check-only
```

The generator must report exact **Decomposition-modified rows**, `multiply` vs `x_over_a` counts, answer bands `100–400` vs `401–999`, and maximum sequence length including padding. Expect **100,000 rows in both arms**, but never assume all are decomposed. Generator never overwrites its output directory.

Outputs:

```text
dpo/data/round5_decomposition/
  original_pairs.json
  decomposition_pairs.json
  manifest.json
```

## GPU Smoke first, then matched experiments

Smoke first:

```bash
sbatch scripts/round5_smoke_decomposition.sbatch
```

After the smoke has **actually completed** (check Slurm logs), submit both full 1-GPU jobs. The TC1 per-user 1-GPU limit means they will normally queue sequentially:

```bash
sbatch scripts/round5_original.sbatch
sbatch scripts/round5_decomposition.sbatch
squeue -u "$USER"
```

Both start from `dpo/experiments/round4_treatment/dpo_epoch1.pt`, 1 epoch, 100k rows, batch 64, LR `1e-5`, seed 42, exact original instructor Course Loss. They train in isolated experiment folders:

- `dpo/experiments/round5_original/`
- `dpo/experiments/round5_decomposition/`

They evaluate (in exactly the same order, never training on these questions): Round3 fixed 768 Validation, old 256 OOD, Round4 fixed 384 extended product Validation, and the Round3 256-entry (potentially duplicated) train probe. No sealed Round3 final Test is evaluated.

After BOTH complete:

```bash
python scripts/compare_round5_decomposition.py
```

**Primary target:** direct product-answer categories, multiply and x_over_a (128/384 extended-val questions). **Guardrail:** original in-domain 768 and old OOD 256; compare paired A-only and B-only counts. The remaining 256 extended multiplication/inverse cases are secondary.

Interpretation requires caution: changing explanatory text also changes token counts and full-sequence loss weighting. If B beats A, consider a subsequent length-matched auxiliary-text control before attributing all gain to decomposition itself.

**Status:** scripts committed, no Round5 dataset generated or GPU training performed by the assistant. Run CPU tests first and share the counts.
