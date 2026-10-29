# SC3000 Overfit-256: instructor loss diagnostic

Purpose: test if the 8.84M NanoGPT can memorize 256 **already-seen** Round 1C incorrect-answer preference pairs. The original course loss, initial `sft/gpt.pt` weights, batch size, and the exact tokenizer are kept. No SFT or reference-model DPO is introduced.

## Run on TC1

```bash
cd ~/NanoGPT-Math
source .venv/bin/activate
git status --short
git pull --ff-only origin main
python scripts/overfit256_course.py --check-data-only
sbatch scripts/overfit256_course.sbatch
```

Do not run `--device cuda` from the login node: use Slurm.

## Protocol

- Reuse the hash-verified Round 1C 20,480-question preference data; deterministic seed 42.
- Select 32 examples per arithmetic category and 16 per equation category: 256 unique seen questions, no validation fact overlap.
- Original `sft/gpt.pt` initialization.
- Same instructor preference loss as Notebook Step 7: `-logsigmoid((pos_logprob-neg_logprob)/beta).mean() - 0.1*pos_logprob.mean()`. Same original `compute_logprob()` from Notebook Step 3. No model changes.
- 128 epochs * 4 steps = 512 optimizer updates. Each of the 256 pairs is seen 128 times, versus 5 times in the 20k run. Learning rate scheduler is unchanged in form, but its total horizon necessarily changes.
- Greedy decoding with original stop rules, 256 seen-train and 256 frozen val questions. Preference win rate before and after.
- Results isolated under `dpo/experiments/overfit256_course/`, summary in `results/overfit256_course_summary.json`; existing Round 1A/B/C checkpoints and data are not modified.

## How to interpret

| Observation | Interpretation / next test |
|---|---|
| Seen-train accuracy climbs substantially | Model can memorize under course loss; large-run underfitting may relate to per-example optimization, coverage, or task complexity. Larger data is still not automatically the solution. |
| Pairwise wins rise but accuracy remains low | Ranking correct over one bad sample is not sufficient to make the correct answer the greedy generation. Test completion-only chosen-answer NLL. |
| Both remain low after 512 updates | Original objective still struggles with tiny data: prioritize answer-masked training objective and optimization audit before a 100k run. |
| Seen-train high, val low | Memorization possible but generalization not yet learned; design simpler curriculum and broader coverage. |

**IMPORTANT:** This diagnostic uses its own short training loop to avoid hundreds of per-epoch checkpoint writes in the original Notebook Step 7. It calls the actual Notebook Steps 2-6, retains their optimizer/scheduler/helpers, and uses the exact Step 7 mathematical loss. It does not claim to execute Step 7 verbatim. The assignment's official Notebook and all previous runs remain untouched. The final submission still needs completed Step 8.
