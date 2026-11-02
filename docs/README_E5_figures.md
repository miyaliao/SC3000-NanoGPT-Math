# Generate the two final Random E5 figures

The plots both evaluate **the same final checkpoint**, `dpo/dpo.pt` (Round9 Random E5). They never use the sealed final test. Historical Round5 / Round7 results are **not** included in the main figures.

## Prerequisites

Run these commands **from the local NanoGPT-Math repository root** (on NTU TC1):

```bash
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install matplotlib numpy
```

The code assumes `model.py`, `sft/meta.pkl`, `dpo/dpo.pt`, and the frozen benchmark data are already present on that machine. In particular, the 768-question validation set and manifest must exist at `dpo/data/round3_structured_full/{val.json,manifest.json}`. These files and the model weights may be intentionally absent from the public GitHub repository; `git pull` does **not** download or recreate them. If `dpo/dpo.pt` is missing, first export the selected Random E5 model using the final notebook Step 8. **Do not substitute an unverified checkpoint.**

First verify that the submitted weight tensors match the archived Random E5 and the development datasets match their frozen SHA256 manifests:

```bash
python scripts/nanogpt_e5_final_figures.py --repo . --preflight
```

## Option A — reuse already saved notebook evaluation (CPU, faster)

If the final notebook Step 8 already produced the `final_reproduction_manifest.json` and `submitted_e5_development` predictions, run:

```bash
python scripts/nanogpt_e5_final_figures.py --repo . --from-notebook
```

This re-scores saved predictions and checks the notebook's recorded export source and tensor identity against `dpo/dpo.pt`. If the notebook exported a fresh retrained checkpoint instead of the historical archive, this workflow will stop unless its identity is independently consistent; see the error message rather than reusing historical results.

## Option B — evaluate the submitted model on one GPU

If the notebook's predictions are not present, submit a job **from the repo root**:

```bash
sbatch scripts/nanogpt_e5_figures.sbatch
```

For direct execution inside an existing Slurm GPU session:

```bash
python scripts/nanogpt_e5_final_figures.py --repo . --device cuda
```

Both options evaluate only **Val768 and Teacher680** with greedy decoding, `max_new_tokens=80`, and the original notebook's strict answer-first integer scorer. No training and no sealed test access occur.

## Output

| File | Meaning |
| --- | --- |
| `assets/figure1_final_e5.png` (`.svg`) | One final E5 checkpoint on Val768 and Teacher680, separately labeled |
| `assets/figure4_e5_skill_errors.png` (`.svg`) | 12-category Teacher680 breakdown of **that same E5**; correct/wrong-answer/format failure |
| `results/portfolio_e5/e5_development_metrics.json` | Checkpoint SHA256, dataset SHA256, category counts and evaluation protocol |
| `results/portfolio_e5/*_predictions.json` | Per-question predictions, when GPU evaluation was used |

To regenerate images from an existing verified local evaluation without inference:

```bash
python scripts/nanogpt_e5_final_figures.py --repo . --plot-only
```

The Notebook's historical selected E5 reference scores are **614/768** on Val768 and **521/680** on Teacher680. The script **does not assume** the scores: it reads saved verified predictions or evaluates the actual `dpo/dpo.pt`. If scores differ, inspect checkpoint provenance and evaluation logs before publishing.

To commit only the resulting final images (not weights, prediction logs, or unverified screenshots):

```bash
git add assets/figure1_final_e5.png assets/figure1_final_e5.svg \
        assets/figure4_e5_skill_errors.png assets/figure4_e5_skill_errors.svg
git commit -m "docs: add verified Random E5 evaluation figures"
git push origin main
```