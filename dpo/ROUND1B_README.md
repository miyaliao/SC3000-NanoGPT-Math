# SC3000 NanoGPT-Math — Round 1B (integer-only ablation)

## Motivation and isolation

Round 1A (20,480 preference pairs, 85% integer + 15% finite decimal) reached 0/256 correct, 0/256 refusals and 256/256 format failures using the *decimal-safe* decoder. Its DPO training achieved low preference loss, but that did **not** improve mathematical accuracy. Round 1B tests whether removing decimals helps under the instructor's original period/newline stopping rule.

**No existing `dpo/dpo.pt` or `dpo/pos_neg_pairs.json` is modified.** The Round 1A checkpoint stays at `dpo/dpo.pt` and the Round 1B checkpoint is saved at `dpo/experiments/round1b_integer/dpo.pt`.

The generator preserves all **17,408 Round 1A integer training examples**, replacing only the **3,072 decimal examples** with newly synthesized integer examples, for 20,480 training pairs total. All 12 categories retain their original quotas: 2,560 samples for each arithmetic type and 1,280 for each equation type. The multiplication replacement examples use a modestly extended 31–50 multiplier because the original (1–100)×(1–30) candidate pool was nearing exhaustion; thus the data differ in *both* numeric type and some operand range. Do not claim this is a perfectly single-variable causal ablation.

Validation: exactly **256 integer questions**, stratified (32 per arithmetic type, 16 per equation type) and sampled with a fixed seed from Round 1A's previously frozen `round1_val.json`. These questions are mathematically disjoint from both training sets. For apples-to-apples scores, evaluate the baseline, Round 1A and Round 1B on precisely this same file with the same evaluator.

## 1. Install scripts and prepare data

```bash
cd ~/NanoGPT-Math
# If scripts were just committed and your working tree is clean:
git pull --ff-only origin main
source .venv/bin/activate
python scripts/generate_round1b.py
python scripts/generate_round1b.py --verify-only
```

Expected SHA-256 of `dpo/experiments/round1b_integer/pos_neg_pairs.json`:

```
0c641927173c7d0e9d221050ded48d273eac9b25da2752b1c6f0569015512164
```

Requires the original Round 1A generator `scripts/generate_round1.py`, manifest and validation file at `dpo/data/round1_full/` to be present. If those are missing, recreate them **without** `--install-train`:

```bash
python scripts/generate_round1.py --profile full
```

Do **not** run the Round 1A training job again. Keep `dpo/dpo.pt` unchanged; it is the model to compare against.

## 2. Train/evaluate on one TC1 GPU

```bash
sbatch scripts/round1b_train.sbatch
squeue -u "$USER"
```

The job performs **four** steps:

1. Baseline (`sft/gpt.pt`) → common integer validation 256
2. Round 1A (`dpo/dpo.pt`) → same common integer validation 256
3. Train Round 1B from **original `sft/gpt.pt`**, executing **the existing notebook Step 2–7 cells** with epochs=5, batch_size=64, 320 steps/epoch
4. Round 1B (`dpo/experiments/round1b_integer/dpo.pt`) → same common integer validation 256

Log files: `round1b_<jobid>.out` and `.err`. Prediction files in `results/`:

- `round1b_common_baseline.jsonl`
- `round1b_common_round1a.jsonl`
- `round1b_common_round1b.jsonl`

The training summary is `results/round1b_training_summary.json`.

**Evaluator compatibility:** `scripts/evaluate_ta_integer.py` uses greedy token selection and the **original model's stop markers** (`\n`: stop without emitting; `.`: emit and stop). It parses integers only. The actual TA scorer's decoding configuration is not public, so this is TA-**like**, not guaranteed identical.

## 3. Read results

```bash
cat results/round1b_training_summary.json

grep -E 'MODEL |overall' round1b_<JOB_ID>.out

python - <<'PY'
import json
from collections import Counter
for key in ('baseline', 'round1a', 'round1b'):
    path=f'results/round1b_common_{key}.jsonl'
    rows=[json.loads(line) for line in open(path)]
    counts=Counter(row['status'] for row in rows)
    print(key, len(rows), dict(counts))
    for row in rows[:5]:print(' ',row['prompt'],row['completion'][:120])
PY
```

**Interpretation:** Compare `correct`, `math_error`, `refusal`, and `format_failure` across three checkpoints under precisely the same conditions. Low training preference loss by itself does not mean arithmetic is correct. If Round 1B is still 0% accurate, test alternative negative construction / answer-only loss as a later **separate** experiment, not by silently modifying this run.

## Deliverables

Generator: `scripts/generate_round1b.py`
Train (Notebook cell execution): `scripts/train_round1b.py`
Original-stop evaluator: `scripts/evaluate_ta_integer.py`
Slurm launcher: `scripts/round1b_train.sbatch`
Reproducibility: `dpo/experiments/round1b_integer/round1b_manifest.json`

Data generator and training data must be included in the eventual submission ZIP if using Round 1B as the final model. The Notebook's Step 8 still needs final completion before submission. Keep the test set sealed until the model-selection method has been chosen.
