# Round 7 — Failure-Pattern Targeted Fine-Tuning (proposal / diagnostic phase)

**Status (2026-10-30): no new Round7 training data or checkpoint yet.** Run the saved-error audit first and use its actual category/difficulty breakdown to decide which training-only skills to reinforce. Do **not** retrain directly on validation failures or the 2025 public GitHub's Blind61/Blind41 questions.

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

## Step 2: Construct two matched *TRAIN-only* data treatments (not implemented yet)

Both arms **start from the identical Round5-A `dpo.pt`**. Keep the current Positive convention `The answer is N because ...` and the original instructor Course Loss. Same batch size, seed, AdamW configuration, learning-rate schedule, steps and number of rows in both arms.

Proposed experiment on **100,000 rows per arm**:
- **Replay-only control:** 70,000 original Round4 Treatment source preference rows (fixed shared across arms), plus 30,000 other old TRAIN records sampled separately. No additional math facts.
- **Targeted skills treatment:** the SAME 70,000 shared original records, plus 15,000 targeted *TRAIN-only* difficult examples from distinct underperforming categories, plus 15,000 **new TRAIN-split math facts and equivalent algebraic forms** representing the same targeted arithmetic skills. Avoid regenerating exact held-out questions or sharing canonical arithmetic fact keys with ANY fixed Val/OOD/Extended/independent Test.
- Avoid answer-only or pre-answer reasoning format changes this round: the variable of interest is **data curriculum**, not response format. Keep both arms identical in positive template and answer extraction convention.
- Candidate targeted skills to confirm or reprioritize from CPU audit: carry/borrow, two-digit times one-digit partial products, multi-digit addition with carry, direct multiplication with output 401–999, and `x/a=b,x=?` direct-product inversion.
- Require SHA256 data manifests, canonical fact verification, tokenizer OOV/256-block-size checks, and train/validation math-fact disjointness for new examples. Ensure both arms have the same old replay rows; report unique prompts, unique math facts and actual category quotas, not just 100k exposure count.

## Step 3: Train/evaluate (only after Step 1 decisions and CPU validation)

A controlled one-epoch pilot is reasonable because the Round5-A model is already capable; use an LR lower than initial SFT training, e.g. `1e-5` as a **provisional** value, and keep it identical in both arms. Restarting optimizer/scheduler is permitted if both arms restart identically.

Primary model selection: **fixed, fact-disjoint** Val768. Guardrails: Old OOD256, ExtendedVal384 and direct product128. Compare *paired* correct-only-in-treatment vs correct-only-in-control counts to avoid claiming gains from simply doing more updates. Public prior-year tests are external diagnostics to be kept out of tuning decisions. If existing Val and OOD are repeatedly used over many iterations, recognize possible validation overfitting; final independent sealed Round3 Test must stay closed until model selection is final.

## Avoid the tempting invalid design

Do **not** take any failed question from `val_predictions.json`, `old_ood_predictions.json`, `extended_val_predictions.json` or `round5_A_greedy_blind*_predictions.json` and insert the exact question or its inverse/commuted mathematical relation into the next training data. That would artificially improve the metric we use to claim generalization.

The 2025 prior-year repository focused on dense deterministic arithmetic patterns and trained more total epochs, but its sampled curated 41-item blind set included at least 20 exact prompts guaranteed by its own deterministic training generator. We should borrow the **teaching curriculum idea** without introducing the same train-test contamination.

**Next action:** run `scripts/audit_round7_failure_patterns.py` on CPU and paste its category/feature breakdown. Then finalize quotas and implement a paired fact-disjoint generator/trainer.
