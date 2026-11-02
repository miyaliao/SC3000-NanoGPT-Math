# SC3000 DPO — Final notebook submission

**Notebook:** `dpo/dpo.ipynb`. The instructor's original **Step 1–8** headings,
core `compute_logprob` and `get_batches`, SFT model load, Step 5 data
validation, Step 6 optimizer configuration and **unchanged Step 8 test_set**
are retained. Only the original training and inference placeholders are
completed/extended.

## Run All modes

- `TRAINING_POLICY="auto"` (default): choose exact **historical Random E5**
  replay when all five frozen training sources (Round3/Round4/Round5/Round7/
  Round9) are present and Random100k matches its known SHA256. In this path,
  train from SFT using the original preference loss, 13 epochs across seven
  stages with optimizer/scheduler restarts and per-epoch checkpoint/optimizer/
  RNG snapshots. E5's historical training weights are selected if available
  and a parity comparison with the newly replayed E5 is recorded.
- Otherwise automatically choose **course mode**, using the mandatory
  `dpo/pos_neg_pairs.json`. When it contains 100,000 rows, use it unchanged.
  If absent/too small, a self-contained verified 12-category arithmetic
  generator in **Step 5** creates 100,000 seeded preference pairs. The original
  Step 7 loop then runs five epochs from SFT, saving `dpo/dpo.pt`.
  This is a runnable course baseline, **not** an exact E5 reproduction.
- `FAST_SMOKE=True`: 128 examples / one epoch, save to a separate smoke
  checkpoint without overwriting `dpo/dpo.pt`. Set it back to `False`
  before final execution.

## Required artifact files

The grading ZIP must include, at minimum:

- `dpo/dpo.pt` — fine-tuned checkpoint
- `dpo/dpo.ipynb` — saved after running from top to bottom, including actual
  data summary, training logs and original Step 8 model outputs
- `dpo/pos_neg_pairs.json` — **100,000** positive/negative rows loaded
  by the instructor's original Step 5

The TA should also have the **course-provided input model and tokenizer**
(`sft/gpt.pt`, `sft/meta.pkl`, `model.py`) for a fresh training run.
For **exact** seven-stage E5 replay, all frozen historical stage data and
associated generator code must also be available; the final Random100k
JSON alone cannot recreate the previous curriculum stages. Additional data
files may require a separate delivery method if the submission ZIP or email
attachment becomes too large.

## Evaluation and writing

Step 8 prints the **original eight instructor prompts** including repeated
prompts. It also calculates real numeric-answer development results when
the frozen Val768 and Seen-heavy Teacher680 datasets exist. No private TA
test is used for model selection. The notebook includes analysis of why E5
was preferred over later control and carry-focused experiments, notes on
repeated mathematical facts, and source attribution for the course-provided
scaffold and external libraries. Complete the contributor/name placeholder
with accurate team information.

## Verification

GitHub Actions checks all notebook Python cells, exact original Step 1–8
headings/test prompts, a deterministic data synthesis generator, and a
disposable **CPU end-to-end smoke** on the original 8.84M GPT architecture.
This CI smoke does **not** assert numerical reproducibility of the trained
historical E5. Run the saved notebook on TC1's prepared CUDA kernel before
submitting, keeping all outputs visible when the file is opened without execution.

The original course notebook is preserved as
`dpo/reference_course_notebook_v1.json`. Other historical training
scripts are unaffected.
