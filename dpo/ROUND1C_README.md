# SC3000 Round 1C — Wrong-answer Negative Ablation

## Rationale

Round 1A (85% integer / 15% decimal, refusal negative) scored 4/256 and Round 1B (100% integer, refusal negative) scored 3/256 on the identical integer validation set. Both learned an output template but not arithmetic. The original model was 0/256 (all refusals). To test whether the **negative type** matters, Round 1C uses **all identical 20,480 prompts and positives as Round 1B**, replacing only `Sorry, I don't know.` with a wrong integer after `The answer is `.

The new incorrect answers have exactly the same sign and text length as the correct answers to limit superficial length/style shortcuts. They are generated with a fixed seed, checked against original equations and arithmetic, and labeled as **synthetic incorrect**, NOT as model-generated errors. The original course preference Loss and `compute_logprob` are unchanged.

## Use on TC1

```bash
cd ~/NanoGPT-Math
source .venv/bin/activate
git status --short
git pull --ff-only origin main
python scripts/generate_round1c.py
python scripts/generate_round1c.py --verify-only
sbatch scripts/round1c_train.sbatch
```

Outputs: `dpo/experiments/round1c_incorrect/pos_neg_pairs.json`, `round1c_train_metadata.json`, `round1c_manifest.json`, same frozen validation JSON, then `dpo.pt` and `results/round1c_training_summary.json`. Evaluation predictions are in `results/round1c_common_round1c.jsonl` and logs in `round1c_<JOB_ID>.out` and `.err`. No Round 1A/B models or data files are overwritten.

The correct answers, question order, all 12 category quotas, batch size 64, 5 epochs, initial `sft/gpt.pt`, AdamW and teacher-recommended loss are held fixed from Round 1B. Initial checkpoint is **not** Round 1B. Only the negative text changes. Evaluation uses the same 256 integer questions and the original period/newline stopping semantics with greedy decoding; call this **TA-like**, not the undisclosed real TA evaluator.

### Inspect results

```bash
grep -E 'ROUND 1C TRAINING COMPLETED|overall|MODEL ' round1c_<JOB_ID>.out
cat results/round1c_training_summary.json
```

If Round 1C still refuses most questions, the incorrect-only pairs may not sufficiently dislodge the initial refusal behavior. If it emits well-formed but incorrect integers, this tells us answer preference alone is insufficient for arithmetic generalization with the 8.84M character model. Neither case implies that the model is broken; it informs whether to try a **mixed negative** experiment next, or study the sequence-mean loss and answer-only masking as an explicitly labeled additional experiment. Keep the original Lab loss as the main comparison.

Caution: do not evaluate or mine from held-out test questions until the protocol is fixed. For final submission, the Notebook Step 8 still needs completion. The generator code and generated preference data must be submitted with the final Notebook and checkpoint.
