# SC3000 Final Submission Notebook — Run All reproduction

**File:** `dpo/dpo.ipynb` on branch `sc3000-final-reproducible-notebook`.

## What the notebook actually reproduces

The selected *historical* checkpoint is Round9 Random E5 (not a direct
SFT→5-epoch fine-tune). Notebook execution follows:

| Stage | Data | Optimizer epochs executed | Cosine scheduler epoch horizon | Learning rate |
| --- | --- | ---: | ---: | ---: |
| SFT → Round3 | 100k Round3 structured | 5 | 5 | 1e-4 |
| Round3 → Round4 Treatment E1 | 100k Treatment | **1** | **2** | 2e-5 |
| Round4 E1 → Round5 Original | 100k one-step explanation | 1 | 1 | 1e-5 |
| Round5 → Round7 Replay | 100k Replay | 1 | 1 | 1e-5 |
| Round7 → Round9 Random E1 | 100k Random | 1 | 1 | 1e-5 |
| Random E1 → Random E2 | 100k Random | 1 (fresh AdamW) | 1 | 1e-5 |
| Random E2 → E3/E4/E5 | 100k Random | **3** | **4** | 1e-5 |

Total: **13 epochs** × 1,562 complete 64-example minibatches = **20,306
optimizer updates**. All stages use instructor original whole-sequence
mean-log-probability preference loss, beta=0.5, chosen NLL bonus=0.1,
AdamW weight decay=0.01, character tokenizer 74, GPT block 256,
original greedy decoding with period/newline stop.

**Selection:** historical E5 Val768=614/768 (79.95%) and
Teacher680=521/680. These numbers are previous observed development
metrics, not guaranteed outcomes on a fresh hardware/version replay.

## Run from TC1 or another prepared CUDA kernel

    cd ~/NanoGPT-Math
    git status --short
    git fetch origin
    git switch --track origin/sc3000-final-reproducible-notebook
    source .venv/bin/activate

Open `dpo/dpo.ipynb` in VS Code/Jupyter with the **NanoGPT-Math
.venv Python kernel**, and use **Restart & Run All**. The default setting
`FAST_SMOKE = False` is the full faithful 13-epoch replay.

**Optional smoke:** Change `FAST_SMOKE = True` in the first code cell,
Run All with the correct Python kernel. Smoke uses 128 rows per stage,
cannot reproduce E5 accuracy, and **never publishes dpo/dpo.pt**.
Change back to False before formal execution.

In a clean checkout the Notebook has 12 exact upstream data-generator
source snapshots embedded as collapsed code cells. It restores missing
files under `scripts/` and checks any present scripts against the
embedded version; it builds missing datasets automatically. The
original course `model.py`, `sft/gpt.pt`, `sft/meta.pkl`, and frozen
teacher/public JSON are input artifacts. Missing data/source/SFT gives
an explicit error rather than inventing a replacement.

**Slurm QoS caveat:** TC1 limits each GPU job to six hours. We cannot
guarantee the complete 13-epoch chain finishes within one such job.
The Notebook saves a full checkpoint/optimizer/RNG/shuffled-order
state after *every complete epoch*. After a job terminates, start a
new allocation, open the same notebook and use Run All again; it checks
and skips finished stages, and resumes the first incomplete stage from
its last complete epoch. A stage that failed before any completed epoch
may be safely restarted if its output directory contains only the
unchanged protocol.json. Never delete archived experiments or frozen
data to fix an output conflict.

## What is submitted

- `dpo/dpo.ipynb`, after **actually running and saving its output
  cells** (a GitHub syntax check does not count as execution).
- `dpo/dpo.pt`: source is **explicit** in
  `dpo/experiments/final_notebook_e5_reproduction/final_reproduction_manifest.json`.
  When the historical E5 archive is present, the default
  `FINAL_EXPORT_POLICY="archived_if_available"` preserves that
  originally selected model; the fresh notebook replay remains a
  separate parity-audited run. When the archive is absent, the trained
  notebook E5 is exported instead. Any weight difference is documented,
  not represented as exact reproduction.
- The full preference data and data-generator code relevant to
  the selected training path, along with provided model.py and SFT
  initialization assets. Most generated `dpo/data/**` is ignored by
  Git; include it explicitly in the final assignment package.
- Frozen Val768 and seen-heavy Teacher680 are for development analysis,
  not the sealed Round3 Test768 or TA hidden test.

## Verification

Notebook source/JSON and the embedded 12 source scripts are checked by
the `SC3000 final notebook contract` GitHub Action (Python 3.10).
**CI tests syntax and source compatibility only; it does not execute
the GPU training or claim archived checkpoint parity.** The notebook
records real newly measured development accuracies and tensor-wise
archived E5 parity after the user runs it.

The original supplied Step2–8 course notebook is preserved under
`dpo/reference_course_notebook_v1.json` (not another `.ipynb`);
legacy experiment scripts use that source, so replacing `dpo/dpo.ipynb`
does not break the older training utilities.
