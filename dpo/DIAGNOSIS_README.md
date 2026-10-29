# Round 1C — Read-only training/generalization and margin diagnosis

This is **not another training run**. It reads the existing `sft/meta.pkl`,
`round1c_manifest.json`, Round 1C pairs/metadata, the fixed 256-question validation
set, and the existing Round 1C `dpo.pt`. It never changes weights, datasets,
Notebook cells, or the final checkpoint.

## On NTU TC1

```bash
cd ~/NanoGPT-Math
git status --short
git pull --ff-only origin main
source .venv/bin/activate
sbatch scripts/diagnose_round1c.sbatch
```

Then replace `<JOB_ID>`:

```bash
grep -E 'TRAIN accuracy:|VAL accuracy:|Preference pair wins:|Mean full-seq margin:|Mean completion-sum margin:|DIAGNOSIS COMPLETE' diagnose1c_<JOB_ID>.out
cat results/diagnosis_round1c/diagnosis_summary.json
```

Produces:

- `results/diagnosis_round1c/diagnosis_summary.json` — metrics with by-category breakdowns
- `results/diagnosis_round1c/train_predictions_256.jsonl` — seen training prompts and generated answers
- `results/diagnosis_round1c/validation_predictions_256.jsonl` — unseen frozen validation prompts
- `results/diagnosis_round1c/train_pair_margins_256.jsonl` — positive/negative teacher-forced probabilities
- `results/diagnosis_round1c/train_sample_256.json` — audit of fixed stratified training sample

Selection: `seed=42`, from Round 1C's 20,480 training samples, with 32 items
from each of the four arithmetic categories and 16 from each of the eight
equation categories. Validation: **the same 256 common integer questions** as
Rounds 1A, 1B, 1C. No held-out test questions are used.

Greedy generation matches the first-period/newline stopping behavior of the
provided `model.py`, without claiming to reproduce unknown TA sampling rules.
The original `compute_logprob`'s teacher-forced whole-sequence mean is
reproduced by counting all nonzero target characters. Padding (ID 0) is
ignored. The script also scores only the completion (including its leading
space, `The answer is ` and final period) after the prompt.

### How to interpret

| Observation | Interpretation / next move |
|---|---|
| High **seen-training** accuracy, low validation | Strong memorization/generalization gap; consider more varied data/sampling (100k). |
| Both accuracy values low, but correct pairs preferred | Pairwise preferences learned, but not enough to generate correct answers; prioritize answer-focused objective or small-set memorization test. |
| Both accuracy and pairwise preference win-rate low | The current loss isn't reliably ranking known positive vs negative answers; prioritize loss/optimization audit and an overfit probe. |
| High validation accuracy | Broaden evaluation, check per-category robustness before larger training. |

The margin is not accuracy. For equal-length pairs with identical prompts and
no dropout, `completion_sum_margin ≈ full_mean_margin × number_of_scored_tokens`.
Therefore positive/negative *ordering* is approximately unchanged by switching
sum vs mean **for these pairs**, though the magnitude/gradient scaling changes.
Likewise, a model can prefer the single chosen answer over the single synthesized
negative under teacher forcing while still generate a third incorrect answer.

Run a different checkpoint without touching the original data:

```bash
python scripts/diagnose_round1c.py --device cuda \
  --checkpoint dpo/experiments/round1b_integer/dpo.pt \
  --out-dir results/diagnosis_round1b_on_1c_pairs
```

Use `sbatch` or `srun` for GPU; do not run CUDA on the TC1 login node.
