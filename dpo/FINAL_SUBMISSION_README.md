# SC3000 Final Notebook — original teacher Steps 1–8

The submission notebook is `dpo/dpo.ipynb` on branch
`sc3000-final-reproducible-notebook`. It was rebuilt from the
unmodified 16-cell instructor reference, preserved as
`dpo/reference_course_notebook_v1.json`.

## Original code preserved

The original Step 2 configuration, Step 3 helper functions,
Step 4 model load, Step 5 preference loading, and Step 6 AdamW/cosine
configuration remain as original code cells, in order.
The original Step 7 single-stage loop is kept beneath an explicit
`RUN_SINGLE_STAGE_DEMO=False` guard so Run All does not accidentally
train an extra five epochs or overwrite `dpo/dpo.pt`. The original
Step 8 `test_set`, including its duplicate test questions, is unchanged;
only the instructor's inference/printing placeholder is filled.

The extra E5 training logic is appended under Step 7 and the supplementary
development evaluation under Step 8. Only E5-required data generator
entrypoints and integrity checks are displayed under Step 5, not 12
embedded source-script copies. The actual generator implementations remain
in `scripts/` and must be included with submission data if required by
the assignment package. All eight instructor Step headings remain in order.

## Historical E5 model chain

SFT → Round3 structured (5 epochs) → Round4 Treatment E1 (one
epoch of two-epoch cosine schedule) → Round5 Original (1 epoch) →
Round7 Replay (1 epoch) → Round9 Random E1 (1 epoch) → Random E2
(1 epoch with fresh AdamW) → Random E3–E5 (three epochs within a
four-epoch continuous cosine schedule).

Total: 13 training epochs, 20,306 optimizer steps across the seven
stages with 100,000 rows, batch64 and original full-sequence Course
Preference Loss. Each completed epoch saves model, optimizer,
scheduler and RNG states. The original Round3 final Test is not read.

## Running

On TC1, use its NanoGPT-Math `.venv` Jupyter kernel; set
`FAST_SMOKE=True` for a tiny pipeline test that **never exports formal
dpo.pt**, and then `False` for full Run All. Completed epochs resume.
The historical selected E5 is kept as the submission checkpoint when
available; independently replayed weights are compared with that
checkpoint, and any numerical discrepancies are disclosed.

The GitHub workflow validates Python syntax, the instructor's eight
headings, original core helper/model/data/optimizer cells and unchanged
teacher Step 8 question list. **No GPU execution is claimed.** Before
submission, rerun the notebook in TC1, save all actual outputs in the
ipynb, verify `dpo/dpo.pt` and include necessary data/generator files.
