# SC3000 NanoGPT-Math — Answer-Focused Loss Ablation

Motivation: Round 1C's full 20,480-pair training reached only 4/256 held-out accuracy; the original course-loss 256-pair overfitting experiment reached 8/256 (3.12%) on the **seen** questions despite achieving 240/256 (93.75%) correct-vs-wrong preference wins. Learning to rank a provided correct answer above a provided incorrect one is not the same as generating the correct answer greedily.

## What changed in `dpo/dpo.ipynb`?

The notebook now has a **switch**, `loss_mode`, in Step 2. The default remains `course`, preserving the exact instructor loss and all past baselines. Set `loss_mode='answer_focused'` to run the additional loss experiment. Steps 3 and 7 now support a precise response-only mask and answer-only teacher-forced NLL. A standalone runner below selects this mode without modifying the stored Round 1A/B/C models or dataset.

For each pair `(x, y+, y-)`, with the same prompt and output format:

```
pos_sum = Σ log Pθ(y+ token | x, previous tokens)  # completion only
neg_sum = Σ log Pθ(y- token | x, previous tokens)  # completion only
Lpref = -mean(logsigmoid((pos_sum - neg_sum) / beta))
Lnumber = -mean(correct numeric-token logprobs)  # digits/sign only
Lcompletion = -mean(correct response-token logprobs)  # answer template + digits + '.'
L = Lpref + 1.0 * Lnumber + 0.25 * Lcompletion
```

`beta=0.5`, the original `AdamW` learning rate and scheduler, and clipping remain as before. **No reference model is introduced and no separate SFT run is performed.** The extra chosen NLL component is teacher-forced auxiliary supervision *inside this training objective*. This is an instructor-loss variant, **not canonical reference-model DPO**. It is limited to Round 1C's fixed integer-answer response format (`<prompt> The answer is <integer>.`). The masks operate on the **shifted next-character targets**, deliberately excluding the question/prompt and padding/newlines (ID 0).

## First: same 256 seen examples, loss-only comparison

The new runner calls `overfit256_course.select_rows(seed=42)` to select **the identical 256 Round 1C pairs**, preserves the 256 held-out common validation examples, begins from `sft/gpt.pt`, and uses 128 epochs × 4 batches = 512 updates at batch size 64. It also reuses the original Notebook Steps 2–6 and computes the identical new loss definition as Notebook Step 7. Unlike executing Step 7 128 times, it saves **one** experiment checkpoint after the optimization loop.

```bash
cd ~/NanoGPT-Math
source .venv/bin/activate
git status --short
git pull --ff-only origin main
python scripts/overfit256_answer_focused.py --check-data-only
sbatch scripts/overfit256_answer_focused.sbatch
```

Use `squeue -u "$USER"` to check job progress, then inspect `answer256_<JOB_ID>.out` and `.err`. Results:

- `dpo/experiments/overfit256_answer_focused/dpo.pt` (ignored by Git)
- `dpo/experiments/overfit256_answer_focused/{train,val}_predictions.json`
- `results/overfit256_answer_focused_summary.json`

The runner refuses to overwrite an existing output directory. It never changes `dpo/dpo.pt`, `dpo/pos_neg_pairs.json`, or the prior Round 1A/B/C/overfit checkpoints.

The **key diagnostic is seen-train accuracy** on exactly 256 previously studied questions. Compare the prior original-loss overfit result **8/256** with the new run. Also report held-out accuracy, pairwise wins, output diversity, and the new loss components; a higher chosen-token likelihood is not, by itself, an improvement in generation quality. If this 256-example test succeeds, run a new 20k checkpoint experiment using the same Notebook mode; otherwise investigate response formatting/learning rather than allocating an entire 100k run.

## Final submission

Keep the course loss mode available for baseline reproduction. Once an improved variant is selected, ensure the final submitted `dpo/dpo.ipynb` has the **intended mode** selected and is fully runnable from top to bottom with saved outputs, contains the generating data or required generator code, and has a complete Step 8. This comparison does **not** replace the final assignment submission.
