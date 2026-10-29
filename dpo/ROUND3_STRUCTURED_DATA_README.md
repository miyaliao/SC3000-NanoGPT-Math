# Round 3 — Structured Arithmetic Data (2026-10-29)

## Objective
Build ~100,000 preference **rows** with verified one-step math explanations,
following the successful data-oriented approach seen in previous SC3000 work.
Prior 20k DPO experiments reached roughly 1–2% held-out accuracy; changing
digit/EOS Loss did not solve the main arithmetic problem. Keep instructor
course Loss fixed and change the data design first.

Reference: https://github.com/JamesRobertJohns/3000

## Positive and negative formats

Positive: `79-7=? The answer is 72 because 79-7 equals 72.`
Positive: `x+55=95,x=? The answer is 40 because 95-55 equals 40.`
Positive: `72-x=34,x=? The answer is 38 because 72-34 equals 38.`
Negative: `<same prompt> Sorry, I don't know.`

All proofs are single-step, exact integer arithmetic; no decimals,
undefined divisions, or ambiguously solved equations such as `0*x=0`.
No boolean logic questions; arithmetic relations act as structural reasoning.
The original model stops at period/newline, so only the final period is used.

## What is generated

- Dense additive facts using 0..100 operands, their commuted forms,
  subtraction inverses and left/right unknown-variable equations.
- Multiplication with results <=400, exact division, and valid reverse forms.
- Higher sampling emphasis on 0/1 identities, small facts, tens, carry/borrow.
- Exactly 100,000 rows in full mode across 12 math categories.
- 768 distinct prompt rows in each of Val and Test (64 per category).
- Fact-level 80%/10%/10% stable SHA-256 split; mathematical inverses and
  commuted versions of the same arithmetic fact cannot cross splits.
- Prior Round1C common-val-256 facts are excluded if that file exists locally.
- Unique prompt and math fact counts, dataset hashes and curriculum tags
  are included in the manifest.

**Important: 100,000 rows do not mean 100,000 unique problems.**
Structured oversampling is deliberate, and its extent is reported.

## Generate on TC1 (CPU only)

```bash
cd ~/NanoGPT-Math
git status --short
git pull --ff-only origin main
source .venv/bin/activate

python scripts/test_round3_structured.py
python scripts/generate_round3_structured.py --profile mini --verify-only
python scripts/generate_round3_structured.py --profile full --verify-only
python scripts/generate_round3_structured.py --profile full
```

Output (new directory, no overwrite):

```text
dpo/data/round3_structured_full/
  pos_neg_pairs.json
  train_metadata.json
  val.json
  test.json
  train_probe_256.json
  manifest.json
```

For a later answer-only control, run with --write-control on initial
generation or create a different --out-dir. This preserves matching prompts.

## Checks

Every pair has the same question prefix, a verified correct answer,
an independently evaluated arithmetic explanation, only course tokenizer
characters, and at most 256 chars including padding. Invalid samples abort.
Val/Test are never used for training; do not tune on Test.

Generation never overwrites dpo/pos_neg_pairs.json, notebook or checkpoints.
Training **has not run**. The first subsequent training must:
  * initialize sft/gpt.pt, not a previously trained model;
  * use the original instructor loss_mode='course';
  * keep run artifacts in a separate experiment directory;
  * score first numerical answer, not just full-string regex;
  * report seen Train, in-domain Val and separate old OOD validation;
  * analyze explanation correctness as a distinct diagnostic.

Note: the existing evaluate_ta_integer.py is anchored to short answers
and will misclassify because explanations. Do not use it for Round 3
without a because-aware numeric evaluator.

Potential limitation: because follows the answer, so it does not serve
as an explicit chain of thought before the answer in autoregressive inference.
Compare because/no-because with identical math facts, not intuition.
